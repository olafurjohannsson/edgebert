use anyhow::{Result, anyhow};
use async_trait::async_trait;
use bytemuck;
use ndarray::{Array2, Array3, s};
use std::sync::Arc;
use wgpu::include_wgsl;
use wgpu::util::DeviceExt;

use crate::traits::{
    Device, Encoder, EncoderArchitecture, EncoderOutput, Model, TransformerConfig,
};
use crate::weights::ModelWeights;
use crate::wgpu_context::WgpuContext;
// Import the new generic components
use crate::gpu_pipeline::{
    GpuAttentionWeights, GpuEncoderPipeline, GpuTransformerLayer
};
use crate::gpu_ops::ffn::GpuFeedForwardWeights;

/// The GPU backend for a generic Transformer Encoder.
/// It holds the GPU-native weights and the generic pipeline to execute them.
pub struct GpuTransformerEncoder {
    pipeline: GpuEncoderPipeline,

    // CPU-side embeddings for the initial lookup.
    word_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,
    token_type_embeddings: Array2<f32>,

    // GPU-side weight buffers, specific to this model instance.
    embedding_norm_weights: (Arc<wgpu::Buffer>, Arc<wgpu::Buffer>),
    layers: Vec<GpuTransformerLayer>,

    // The config must be wrapped in an Arc to be thread-safe (`Send + Sync`).
    // This allows it to be shared across threads if the model is used in an async context.
    config: Arc<dyn EncoderArchitecture + Send + Sync>,
}

impl GpuTransformerEncoder {
    /// Constructs a new `GpuTransformerEncoder`.
    /// The `C` generic type now has `'static`, `Send`, and `Sync` bounds to ensure thread safety.
    pub fn new<C>(weights: &ModelWeights, config: Arc<C>, context: Arc<WgpuContext>) -> Result<Self>
    where
        C: EncoderArchitecture + Send + Sync + 'static,
    {
        let pipeline = GpuEncoderPipeline::new(context.clone())?;

        let device = &context.device;

        // Helper functions
        let upload_1d = |name: &str| -> Result<Arc<wgpu::Buffer>> {
            let tensor = weights.get_array1(name)?;
            Ok(Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytemuck::cast_slice(tensor.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
                },
            )))
        };

        /// Helper to upload a 2D tensor from `ModelWeights` to a `wgpu::Buffer`.
        let upload_2d = |name: &str| -> Result<Arc<wgpu::Buffer>> {
            // 1. Load the tensor.
            let tensor = weights.get_array2(name)?;

            // 2. Transpose it to match the CPU's data format.
            let transposed_tensor = tensor.t().to_owned();

            // 3. Upload the *transposed* tensor.
            Ok(Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytemuck::cast_slice(
                        transposed_tensor.as_standard_layout().as_slice().unwrap(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            )))
        };

        // Load CPU embeddings
        let (word_w, pos_w, type_w) = config.get_embedding_weight_names();
        let word_embeddings = weights.get_array2(word_w)?;
        let position_embeddings = weights.get_array2(pos_w)?;
        let token_type_embeddings = weights.get_array2(type_w)?;

        // Upload embedding norm weights
        let (norm_w, norm_b) = config.get_embedding_layer_norm_names();
        let embedding_norm_weights = (upload_1d(norm_w)?, upload_1d(norm_b)?);

        // Upload layer weights
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i);
            let ffn_names = config.get_feed_forward_names(i);

            let attention_weights = GpuAttentionWeights {
                q_weight: upload_2d(&attn_names.q_weight)?,
                q_bias: upload_1d(&attn_names.q_bias)?,
                k_weight: upload_2d(&attn_names.k_weight)?,
                k_bias: upload_1d(&attn_names.k_bias)?,
                v_weight: upload_2d(&attn_names.v_weight)?,
                v_bias: upload_1d(&attn_names.v_bias)?,
                output_weight: upload_2d(&attn_names.output_weight)?,
                output_bias: upload_1d(&attn_names.output_bias)?,
                norm_weight: upload_1d(&attn_names.norm_weight)?,
                norm_bias: upload_1d(&attn_names.norm_bias)?,
            };

            let intermediate_w = weights.get_array2(&ffn_names.intermediate_weight)?;
            let intermediate_b = weights.get_array1(&ffn_names.intermediate_bias)?;
            let output_w = weights.get_array2(&ffn_names.output_weight)?;
            let output_b = weights.get_array1(&ffn_names.output_bias)?;

            let mut packed_ffn_data: Vec<f32> = Vec::new();
            if config.transpose_ffn_weights() {
                // For BERT: Transpose [out, in] -> [in, out] to match the shader's expectation.
                let intermediate_w_t = intermediate_w.t().as_standard_layout().to_owned();
                let output_w_t = output_w.t().as_standard_layout().to_owned();
                
                packed_ffn_data.extend_from_slice(intermediate_w_t.as_slice().unwrap());
                packed_ffn_data.extend_from_slice(intermediate_b.as_slice().unwrap());
                packed_ffn_data.extend_from_slice(output_w_t.as_slice().unwrap());
                packed_ffn_data.extend_from_slice(output_b.as_slice().unwrap());
            } else {
                // For GPT-2 style models: Use weights as-is.
                packed_ffn_data.extend_from_slice(intermediate_w.as_standard_layout().as_slice().unwrap());
                packed_ffn_data.extend_from_slice(intermediate_b.as_slice().unwrap());
                packed_ffn_data.extend_from_slice(output_w.as_standard_layout().as_slice().unwrap());
                packed_ffn_data.extend_from_slice(output_b.as_slice().unwrap());
            }
            // packed_ffn_data
            //     .extend_from_slice(intermediate_w.as_standard_layout().as_slice().unwrap());
            // packed_ffn_data.extend_from_slice(intermediate_b.as_slice().unwrap());
            // packed_ffn_data.extend_from_slice(output_w.as_standard_layout().as_slice().unwrap());
            // packed_ffn_data.extend_from_slice(output_b.as_slice().unwrap());

            let packed_weights = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("FFN Packed Weights Layer {}", i)),
                    contents: bytemuck::cast_slice(&packed_ffn_data),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));

            let ffn_weights = GpuFeedForwardWeights {
                packed_weights,
                norm_weight: upload_1d(&ffn_names.norm_weight)?,
                norm_bias: upload_1d(&ffn_names.norm_bias)?,
            };

            // let ffn_weights = GpuFeedForwardWeights {
            //     intermediate_weight: upload_2d(&ffn_names.intermediate_weight)?,
            //     intermediate_bias: upload_1d(&ffn_names.intermediate_bias)?,
            //     output_weight: upload_2d(&ffn_names.output_weight)?,
            //     output_bias: upload_1d(&ffn_names.output_bias)?,
            //     norm_weight: upload_1d(&ffn_names.norm_weight)?,
            //     norm_bias: upload_1d(&ffn_names.norm_bias)?,
            // };

            layers.push(GpuTransformerLayer {
                attention_weights,
                ffn_weights,
            });
        }

        Ok(Self {
            pipeline,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embedding_norm_weights,
            layers,
            config, // Store the config as a thread-safe trait object
        })
    }

    fn perform_cpu_embedding(&self, input_ids: &Array2<f32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.config.hidden_size();
        let mut cpu_hidden_states = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));

        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                if token_id < self.word_embeddings.shape()[0] {
                    cpu_hidden_states
                        .slice_mut(s![i, j, ..])
                        .assign(&self.word_embeddings.row(token_id));
                }
            }
        }
        let pos_embeddings_to_add = self.position_embeddings.slice(s![0..seq_len, ..]);
        cpu_hidden_states += &pos_embeddings_to_add;
        if self.token_type_embeddings.shape()[0] > 0 {
            cpu_hidden_states += &self.token_type_embeddings.row(0);
        }
        Ok(cpu_hidden_states)
    }
}

// NOW THIS COMPILES: The `config` field is `Arc`, which is `Send + Sync`.
impl Model for GpuTransformerEncoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
}

#[async_trait]
impl Encoder for GpuTransformerEncoder {
    type Input = Array2<f32>;
    type Output = EncoderOutput;

    async fn forward(
        &self,
        input_ids: &Self::Input,
        attention_mask: &Array2<f32>,
    ) -> Result<Self::Output> {
        // Step 1: CPU-Side Embedding
        let initial_embeddings = self.perform_cpu_embedding(input_ids)?;

        // Step 2: Call the Generic Pipeline
        let last_hidden_state = self
            .pipeline
            .forward(
                // CORRECTED: Pass the Arc as a reference. `as_ref()` gets a `&dyn Trait`.
                self.config.as_ref(),
                &initial_embeddings,
                &attention_mask,
                (
                    &self.embedding_norm_weights.0,
                    &self.embedding_norm_weights.1,
                ),
                &self.layers,
            )
            .await?;

        Ok(EncoderOutput { last_hidden_state })
    }
}
