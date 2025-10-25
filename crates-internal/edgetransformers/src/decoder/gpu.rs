use anyhow::{Result, anyhow};
use async_trait::async_trait;
use bytemuck;
use ndarray::{Array2, Array3, s};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::gpu_ops::ffn::GpuFeedForwardWeights;
use crate::gpu_pipeline::{GpuAttentionWeights, GpuEncoderPipeline, GpuTransformerLayer};
use crate::traits::{
    Cache, Decoder, DecoderArchitecture, DecoderOutput, Device, Model, TransformerConfig,
};
use crate::weights::ModelWeights;
use crate::wgpu_context::WgpuContext;

/// The GPU backend for a generic Transformer Decoder.
pub struct GpuTransformerDecoder {
    pipeline: GpuEncoderPipeline,

    // CPU-side embeddings
    word_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,

    // GPU-side weight buffers
    final_layer_norm_weights: (Arc<wgpu::Buffer>, Arc<wgpu::Buffer>), // Changed name
    layers: Vec<GpuTransformerLayer>,

    config: Arc<dyn DecoderArchitecture + Send + Sync>,
}

impl GpuTransformerDecoder {
    pub fn new<C>(weights: &ModelWeights, config: Arc<C>, context: Arc<WgpuContext>) -> Result<Self>
    where
        C: DecoderArchitecture + Send + Sync + 'static,
    {
        let pipeline = GpuEncoderPipeline::new(context.clone())?;
        let device = &context.device;

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

        let upload_2d = |name: &str| -> Result<Arc<wgpu::Buffer>> {
            let tensor = weights.get_array2(name)?;
            let transposed_tensor = tensor.t().to_owned();
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

        // Load embeddings (no token_type for decoder)
        let (word_w, pos_w) = config.get_embedding_weight_names();
        let word_embeddings = weights.get_array2(word_w)?;
        let position_embeddings = weights.get_array2(pos_w)?;

        // Upload final layer norm weights (not embedding layer norm!)
        let (norm_w, norm_b) = config.get_final_layer_norm_names();
        let final_layer_norm_weights = (upload_1d(norm_w)?, upload_1d(norm_b)?);

        // Upload layer weights
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i); // Returns LayerDecoderAttentionNames
            let ffn_names = config.get_feed_forward_names(i);

            // For GPT-2: QKV is combined, need to split it
            let qkv_weight = weights.get_array2(&attn_names.qkv_weight)?;
            let qkv_bias = weights.get_array1(&attn_names.qkv_bias)?;

            let hidden_size = config.hidden_size();

            // Split combined QKV into separate Q, K, V for the attention weights struct
            // let qkv_weight_t = qkv_weight.t().to_owned();
            let q_weight = qkv_weight.slice(s![.., 0..hidden_size]).to_owned();
            let k_weight = qkv_weight
                .slice(s![.., hidden_size..2 * hidden_size])
                .to_owned();
            let v_weight = qkv_weight
                .slice(s![.., 2 * hidden_size..3 * hidden_size])
                .to_owned();

            let q_weight_t = q_weight.t().to_owned();
            let k_weight_t = k_weight.t().to_owned();
            let v_weight_t = v_weight.t().to_owned();

            let q_bias = qkv_bias.slice(s![0..hidden_size]).to_owned();
            let k_bias = qkv_bias.slice(s![hidden_size..2 * hidden_size]).to_owned();
            let v_bias = qkv_bias
                .slice(s![2 * hidden_size..3 * hidden_size])
                .to_owned();

            // Upload split Q, K, V weights
            let q_weight_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} Q Weight", i)),
                    contents: bytemuck::cast_slice(
                        q_weight_t.as_standard_layout().as_slice().unwrap(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));
            let k_weight_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} K Weight", i)),
                    contents: bytemuck::cast_slice(
                        k_weight_t.as_standard_layout().as_slice().unwrap(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));
            let v_weight_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} V Weight", i)),
                    contents: bytemuck::cast_slice(
                        v_weight_t.as_standard_layout().as_slice().unwrap(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));
            let q_bias_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} Q Bias", i)),
                    contents: bytemuck::cast_slice(q_bias.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
                },
            ));
            let k_bias_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} K Bias", i)),
                    contents: bytemuck::cast_slice(k_bias.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
                },
            ));
            let v_bias_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} V Bias", i)),
                    contents: bytemuck::cast_slice(v_bias.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
                },
            ));

            let attention_weights = GpuAttentionWeights {
                q_weight: q_weight_buf,
                q_bias: q_bias_buf,
                k_weight: k_weight_buf,
                k_bias: k_bias_buf,
                v_weight: v_weight_buf,
                v_bias: v_bias_buf,
                output_weight: upload_2d(&attn_names.output_weight)?,
                output_bias: upload_1d(&attn_names.output_bias)?,
                norm_weight: upload_1d(&attn_names.norm_weight)?,
                norm_bias: upload_1d(&attn_names.norm_bias)?,
            };

            let intermediate_w = weights.get_array2(&ffn_names.intermediate_weight)?;
            let intermediate_b = weights.get_array1(&ffn_names.intermediate_bias)?;
            let output_w = weights.get_array2(&ffn_names.output_weight)?;
            let output_b = weights.get_array1(&ffn_names.output_bias)?;
            // Transpose if needed (for BERT-style models)
            let fc1_weight_data = if config.transpose_ffn_weights() {
                intermediate_w.t().as_standard_layout().to_owned()
            } else {
                intermediate_w.as_standard_layout().to_owned()
            };

            let fc2_weight_data = if config.transpose_ffn_weights() {
                output_w.t().as_standard_layout().to_owned()
            } else {
                output_w.as_standard_layout().to_owned()
            };

            // Upload FC1 weights
            let fc1_weight_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} FC1 Weight", i)),
                    contents: bytemuck::cast_slice(fc1_weight_data.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));

            let fc1_bias_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} FC1 Bias", i)),
                    contents: bytemuck::cast_slice(intermediate_b.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));

            // Upload FC2 weights
            let fc2_weight_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} FC2 Weight", i)),
                    contents: bytemuck::cast_slice(fc2_weight_data.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));

            let fc2_bias_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} FC2 Bias", i)),
                    contents: bytemuck::cast_slice(output_b.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));

            let ffn_weights = GpuFeedForwardWeights {
                fc1_weight: fc1_weight_buf,
                fc1_bias: fc1_bias_buf,
                fc2_weight: fc2_weight_buf,
                fc2_bias: fc2_bias_buf,
                norm_weight: upload_1d(&ffn_names.norm_weight)?,
                norm_bias: upload_1d(&ffn_names.norm_bias)?,
            };
            // let mut packed_ffn_data: Vec<f32> = Vec::new();
            // if config.transpose_ffn_weights() {
            //     let intermediate_w_t = intermediate_w.t().as_standard_layout().to_owned();
            //     let output_w_t = output_w.t().as_standard_layout().to_owned();
            //     packed_ffn_data.extend_from_slice(intermediate_w_t.as_slice().unwrap());
            //     packed_ffn_data.extend_from_slice(intermediate_b.as_slice().unwrap());
            //     packed_ffn_data.extend_from_slice(output_w_t.as_slice().unwrap());
            //     packed_ffn_data.extend_from_slice(output_b.as_slice().unwrap());
            // } else {
            //     packed_ffn_data.extend_from_slice(intermediate_w.as_standard_layout().as_slice().unwrap());
            //     packed_ffn_data.extend_from_slice(intermediate_b.as_slice().unwrap());
            //     packed_ffn_data.extend_from_slice(output_w.as_standard_layout().as_slice().unwrap());
            //     packed_ffn_data.extend_from_slice(output_b.as_slice().unwrap());
            // }

            // let packed_weights = Arc::new(device.create_buffer_init(
            //     &wgpu::util::BufferInitDescriptor {
            //         label: Some(&format!("FFN Packed Weights Layer {}", i)),
            //         contents: bytemuck::cast_slice(&packed_ffn_data),
            //         usage: wgpu::BufferUsages::STORAGE,
            //     },
            // ));

            // let ffn_weights = GpuFeedForwardWeights {
            //     packed_weights,
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
            final_layer_norm_weights,
            layers,
            config,
        })
    }

    fn perform_cpu_embedding(
        &self,
        input_ids: &Array2<f32>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.config.hidden_size();
        let mut cpu_hidden_states = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));

        // Word embeddings
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

        // Position embeddings (offset by cache position for incremental decoding)
        let pos_start = position_offset;
        let pos_end = position_offset + seq_len;
        let pos_embeddings_to_add = self.position_embeddings.slice(s![pos_start..pos_end, ..]);
        cpu_hidden_states += &pos_embeddings_to_add;

        Ok(cpu_hidden_states)
    }
}

impl Model for GpuTransformerDecoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
}

#[async_trait]
impl Decoder for GpuTransformerDecoder {
    type Input = Array2<f32>;
    type Output = DecoderOutput;

    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        let position_offset = if let Some(ref cache) = cache {
            cache.get_seq_length()
        } else {
            0
        };

        // Step 1: CPU-Side Embedding
        let initial_embeddings = self.perform_cpu_embedding(input, position_offset)?;

        // Step 2: Forward through pipeline (no embedding layer norm for GPT-2!)
        // TODO: Pass empty/identity norm for first layer, use final norm at end
        let last_hidden_state = self
            .pipeline
            .forward(
                self.config.as_ref(),
                &initial_embeddings,
                attention_mask,
                (
                    &self.final_layer_norm_weights.0, // This should be applied at END, not beginning
                    &self.final_layer_norm_weights.1,
                ),
                &self.layers,
            )
            .await?;

        // TODO: Apply final layer norm here instead of in pipeline
        // TODO: Extract and store K, V tensors in cache

        Ok(DecoderOutput {
            last_hidden_state,
            past_key_values: None, // TODO: populate from cache
        })
    }
}
