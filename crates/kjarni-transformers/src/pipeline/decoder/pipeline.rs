use crate::LoadedEmbeddings;
use crate::WgpuContext;
use crate::decoder::prelude::{CpuDecoder, GpuDecoder};
use crate::execution::ExecutionPlan;
use crate::loaders::LoadedLMHead;
use crate::prelude::Device;
use anyhow::{Result, anyhow};
use std::sync::Arc;

pub struct DecoderPipeline {
    embeddings: LoadedEmbeddings,
    cpu_decoder: Option<Box<dyn CpuDecoder>>,
    gpu_decoder: Option<Box<dyn GpuDecoder>>,
    lm_head: LoadedLMHead,
    plan: ExecutionPlan,
    context: Option<Arc<WgpuContext>>,
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    max_sequence_length: Option<usize>,
}

/// Builder configuration for DecoderPipeline
pub struct DecoderPipelineConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_sequence_length: Option<usize>,
    pub max_batch_size: Option<usize>,
}

impl DecoderPipeline {
    /// Creates a new decoder pipeline.
    pub fn new(
        embeddings: LoadedEmbeddings,
        cpu_decoder: Option<Box<dyn CpuDecoder>>,
        gpu_decoder: Option<Box<dyn GpuDecoder>>,
        lm_head: LoadedLMHead,
        plan: ExecutionPlan,
        context: Option<Arc<WgpuContext>>,
        config: DecoderPipelineConfig,
    ) -> Result<Self> {
        let pipeline = Self {
            embeddings,
            cpu_decoder,
            gpu_decoder,
            lm_head,
            plan,
            context,
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            max_sequence_length: config.max_sequence_length,
        };

        // Validate the plan against available components
        pipeline.validate_plan(&pipeline.plan)?;

        Ok(pipeline)
    }
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Update the execution plan.
    pub fn set_plan(&mut self, plan: ExecutionPlan) -> Result<()> {
        self.validate_plan(&plan)?;
        self.plan = plan;
        Ok(())
    }

    fn validate_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        // Validate embeddings
        match plan.embeddings {
            Device::Cpu if !self.embeddings.is_cpu() => {
                return Err(anyhow!("Plan requires CPU embeddings but not loaded"));
            }
            // Native-only check: no wasm plan can select Wgpu.
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
            Device::Wgpu if !self.embeddings.is_gpu() => {
                return Err(anyhow!("Plan requires GPU embeddings but not loaded"));
            }
            _ => {}
        }

        // Validate layers
        match plan.layers {
            Device::Cpu if self.cpu_decoder.is_none() => {
                return Err(anyhow!("Plan requires CPU decoder but not loaded"));
            }
            Device::Wgpu if self.gpu_decoder.is_none() => {
                return Err(anyhow!("Plan requires GPU decoder but not loaded"));
            }
            _ => {}
        }

        // Validate LM head
        match plan.lm_head {
            Device::Cpu if !self.lm_head.has_cpu() => {
                return Err(anyhow!("Plan requires CPU LM head but not loaded"));
            }
            // Native-only check: no wasm plan can select Wgpu.
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
            Device::Wgpu if !self.lm_head.has_gpu() => {
                return Err(anyhow!("Plan requires GPU LM head but not loaded"));
            }
            _ => {}
        }

        // Validate GPU context if needed
        if plan.needs_gpu() && self.context.is_none() {
            return Err(anyhow!("Plan requires GPU but no WgpuContext available"));
        }

        Ok(())
    }

    pub fn embeddings(&self) -> &LoadedEmbeddings {
        &self.embeddings
    }

    pub fn cpu_decoder(&self) -> Option<&dyn CpuDecoder> {
        self.cpu_decoder.as_ref().map(|d| d.as_ref())
    }

    pub fn gpu_decoder(&self) -> Option<&dyn GpuDecoder> {
        self.gpu_decoder.as_ref().map(|d| d.as_ref())
    }

    pub fn lm_head(&self) -> &LoadedLMHead {
        &self.lm_head
    }

    pub fn context(&self) -> Option<&Arc<WgpuContext>> {
        self.context.as_ref()
    }
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn max_sequence_length(&self) -> Option<usize> {
        self.max_sequence_length
    }

    pub fn max_batch_size(&self) -> Option<usize> {
        None
    }
    /// Get the active decoder based on the current plan.
    pub fn active_cpu_decoder(&self) -> Result<&dyn CpuDecoder> {
        self.cpu_decoder
            .as_ref()
            .map(|d| d.as_ref())
            .ok_or_else(|| anyhow!("CPU decoder not available"))
    }

    /// Get the active GPU decoder based on the current plan.
    pub fn active_gpu_decoder(&self) -> Result<&dyn GpuDecoder> {
        self.gpu_decoder
            .as_ref()
            .map(|d| d.as_ref())
            .ok_or_else(|| anyhow!("GPU decoder not available"))
    }
}

#[cfg(test)]
mod decoder_pipeline_test {
    use ndarray::{Array2, Array3};

    use super::*;
    use crate::gpu::{GpuTensor, GpuTensorPool};
    use crate::models::base::{ModelInput, ModelLoadConfig};
    use crate::pipeline::DecoderPipelineBuilder;
    use crate::tensor::DType;
    use crate::traits::{ModelLayout, ModelMetadata};
    use crate::weights::ModelWeights;
    use crate::{Cache, WgpuContext, gpu::cache::GpuKVCache};
    use std::sync::Arc;

    struct MockCpuDecoder {
        num_layers: usize,
    }
    impl CpuDecoder for MockCpuDecoder {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn final_norm(&self, _: &Array3<f32>) -> Result<Array3<f32>> {
            unimplemented!()
        }
        fn head_dim(&self) -> usize {
            0
        }
        fn hidden_size(&self) -> usize {
            0
        }
        fn num_kv_heads(&self) -> usize {
            0
        }
        fn num_attention_heads(&self) -> usize {
            0
        }

        fn forward_layers(
            &self,
            _: &Array3<f32>,
            _: &Array2<f32>,
            _: usize,
            _: Option<&mut dyn Cache>,
            _: usize,
            _: usize,
        ) -> anyhow::Result<Array3<f32>> {
            Ok(Array3::zeros((1, 1, 1)))
        }
        fn num_layers(&self) -> usize {
            self.num_layers
        }
    }

    struct MockGpuDecoder {
        num_layers: usize,
    }
    impl GpuDecoder for MockGpuDecoder {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn embed(
            &self,
            _encoder: &mut wgpu::CommandEncoder,
            _pool: &mut GpuTensorPool,
            _input: ModelInput<'_>,
            _position_offset: usize,
        ) -> Result<GpuTensor> {
            unimplemented!()
        }
        /// Metadata: Hidden dimension size.
        fn hidden_size(&self) -> usize {
            0
        }
        fn embed_and_normalize(
            &self,
            _encoder: &mut wgpu::CommandEncoder,
            _pool: &mut GpuTensorPool,
            _input: ModelInput<'_>,
            _position_offset: usize,
        ) -> Result<GpuTensor> {
            unimplemented!()
        }

        fn forward_layers(
            &self,
            _: &mut wgpu::CommandEncoder,
            _: &mut GpuTensorPool,
            _: &GpuTensor,
            _: &GpuTensor,
            _: usize,
            _: Option<&mut GpuKVCache>,
            _: usize,
            _: usize,
        ) -> anyhow::Result<GpuTensor> {
            unimplemented!()
        }
        fn num_layers(&self) -> usize {
            self.num_layers
        }
    }

    use crate::traits::{AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout};

    struct MockConfig {
        hidden: usize,
        vocab: usize,
        tied: bool,
    }

    impl crate::traits::ModelConfig for MockConfig {
        fn model_type(&self) -> &str {
            "mock_pipeline"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn metadata(&self) -> ModelMetadata {
            ModelMetadata {
                decoder_layers: None,
                intermediate_size: 0,
                hidden_size: self.hidden,
                num_layers: 1,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: self.hidden,
                vocab_size: self.vocab,
                max_seq_len: 128,
                norm_eps: 1e-5,
                activation: crate::activations::Activation::Gelu,
                rope_theta: None,
                rope_scaling: None,
                scale_embeddings: false,
                normalize_embedding: false,
                extra_pos_embeddings: 0,
                is_prenorm: true,
                transpose_ffn_weights: false,
                transpose_attention_weights: false,
                problem_type: None,
                normalization_strategy: crate::traits::NormalizationStrategy::RMSNorm,
                no_scale_qk: false,
            }
        }

        fn layout(&self) -> ModelLayout {
            let layer_layout = DecoderLayerLayout {
                self_attn: AttentionLayout {
                    q_weight: "layer.{}.q.weight".to_string(),
                    k_weight: "layer.{}.k.weight".to_string(),
                    v_weight: "layer.{}.v.weight".to_string(),
                    o_weight: "layer.{}.o.weight".to_string(),
                    norm_weight: "layer.{}.attn_norm.weight".to_string(),
                    q_bias: None,
                    k_bias: None,
                    v_bias: None,
                    o_bias: None,
                    norm_bias: None,
                },
                cross_attn: None,
                ffn: FeedForwardLayout {
                    gate_weight: Some("layer.{}.gate.weight".to_string()),
                    up_weight: "layer.{}.up.weight".to_string(),
                    down_weight: "layer.{}.down.weight".to_string(),
                    norm_weight: "layer.{}.ffn_norm.weight".to_string(),
                    gate_bias: None,
                    up_bias: None,
                    down_bias: None,
                    norm_bias: None,
                },
            };

            ModelLayout {
                token_embedding: "token_embd.weight".to_string(),
                lm_head: if self.tied {
                    "token_embd.weight".to_string()
                } else {
                    "output.weight".to_string()
                },
                encoder: None,
                decoder: Some(DecoderLayout {
                    position_embedding: None,
                    token_type_embedding: None,
                    embedding_norm_weight: None,
                    embedding_norm_bias: None,
                    final_norm_weight: Some("norm.weight".to_string()),
                    final_norm_bias: None,
                    layer: layer_layout,
                }),
            }
        }
    }
    fn get_mock_backends() -> (Option<Box<dyn CpuDecoder>>, Option<Box<dyn GpuDecoder>>) {
        (
            Some(Box::new(MockCpuDecoder { num_layers: 1 })),
            Some(Box::new(MockGpuDecoder { num_layers: 1 })),
        )
    }
    fn create_dummy_weights(
        tensors: Vec<(&str, Vec<f32>, Vec<usize>)>,
    ) -> (tempfile::TempDir, ModelWeights) {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let dir = tempfile::TempDir::new().unwrap();
        let model_path = dir.path().join("model.safetensors");
        let config_json = r#"{ "hidden_size": 32, "vocab_size": 100 }"#;
        std::fs::write(dir.path().join("config.json"), config_json).unwrap();

        let mut data_map = std::collections::HashMap::new();
        let mut buffers = Vec::new(); // Keep alive

        for (name, data, shape) in tensors {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            buffers.push((name.to_string(), bytes, shape));
        }
        for (name, bytes, shape) in &buffers {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap();
            data_map.insert(name.clone(), view);
        }

        let serialized = serialize(&data_map, &None).unwrap();
        std::fs::write(&model_path, &serialized).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        (dir, weights)
    }

    #[test]
    fn test_builder_cpu_plan_tied_weights() {
        let hidden = 32;
        let vocab = 100;

        let (_dir, weights) = create_dummy_weights(vec![(
            "token_embd.weight",
            vec![1.0; vocab * hidden],
            vec![vocab, hidden],
        )]);

        let config = Arc::new(MockConfig {
            hidden,
            vocab,
            tied: true,
        });

        // Default config = CPU
        let load_config = ModelLoadConfig::default();
        let (cpu, gpu) = get_mock_backends();
        let pipeline = DecoderPipelineBuilder::new(&weights, config)
            .with_load_config(load_config)
            .with_backends(cpu, gpu)
            .build()
            .expect("Failed to build pipeline");

        // Verify Embeddings Loaded
        assert!(pipeline.embeddings().is_cpu());

        // Verify LM Head Loaded
        assert!(pipeline.lm_head().has_cpu());

        // Verify Sharing (Pointer Equality)
        let emb_ptr = pipeline.embeddings().word_embeddings_cpu().unwrap().data;
        let head_ptr = pipeline
            .lm_head()
            .cpu_weights
            .as_ref()
            .unwrap()
            .data
            .clone();

        // Need to match on LinearData variants to check Arcs
        match (emb_ptr, head_ptr) {
            (
                crate::linear_layer::LinearData::F32(arc1),
                crate::linear_layer::LinearData::F32(arc2),
            ) => {
                assert!(
                    Arc::ptr_eq(&arc1, &arc2),
                    "Weights should be shared via Arc"
                );
            }
            _ => panic!("Expected F32 weights"),
        }
    }

    #[test]
    fn test_builder_cpu_plan_untied_weights() {
        let hidden = 32;
        let vocab = 100;

        let (_dir, weights) = create_dummy_weights(vec![
            (
                "token_embd.weight",
                vec![1.0; vocab * hidden],
                vec![vocab, hidden],
            ),
            (
                "output.weight",
                vec![2.0; vocab * hidden],
                vec![vocab, hidden],
            ),
        ]);

        let config = Arc::new(MockConfig {
            hidden,
            vocab,
            tied: false,
        });
        let (cpu, gpu) = get_mock_backends();
        let pipeline = DecoderPipelineBuilder::new(&weights, config)
            .with_backends(cpu, gpu)
            .build()
            .expect("Failed to build pipeline");

        let emb_val = pipeline.embeddings().word_embeddings_cpu().unwrap().data;
        let head_val = pipeline
            .lm_head()
            .cpu_weights
            .as_ref()
            .unwrap()
            .data
            .clone();

        match (emb_val, head_val) {
            (
                crate::linear_layer::LinearData::F32(arc1),
                crate::linear_layer::LinearData::F32(arc2),
            ) => {
                // Should NOT be equal
                assert!(!Arc::ptr_eq(&arc1, &arc2));
                assert_eq!(arc1[[0, 0]], 1.0);
                assert_eq!(arc2[[0, 0]], 2.0);
            }
            _ => panic!("Expected F32 weights"),
        }
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_builder_gpu_plan() -> Result<()> {
        let context = WgpuContext::new().await?;
        let hidden = 32;
        let vocab = 100;

        let (_dir, weights) = create_dummy_weights(vec![(
            "token_embd.weight",
            vec![1.0; vocab * hidden],
            vec![vocab, hidden],
        )]);

        let config = Arc::new(MockConfig {
            hidden,
            vocab,
            tied: true,
        });
        let (cpu, gpu) = get_mock_backends();
        let pipeline = DecoderPipelineBuilder::new(&weights, config)
            .with_backends(cpu, gpu)
            .with_context_opt(Some(context))
            .build()?;
        assert!(pipeline.plan().embeddings.is_gpu());
        assert!(pipeline.plan().lm_head.is_gpu());

        assert!(pipeline.embeddings().is_gpu());
        assert!(pipeline.lm_head().has_gpu());
        Ok(())
    }

    #[test]
    fn test_builder_explicit_offload() {
        let hidden = 32;
        let vocab = 100;
        let (_dir, weights) = create_dummy_weights(vec![(
            "token_embd.weight",
            vec![1.0; vocab * hidden],
            vec![vocab, hidden],
        )]);
        let config = Arc::new(MockConfig {
            hidden,
            vocab,
            tied: true,
        });

        let load_config = ModelLoadConfig {
            offload_embeddings: true, // Force CPU embeddings
            ..Default::default()
        };
        let (cpu, gpu) = get_mock_backends();
        let pipeline = DecoderPipelineBuilder::new(&weights, config)
            .with_backends(cpu, gpu)
            .with_load_config(load_config)
            .build()
            .unwrap();

        assert!(pipeline.embeddings().is_cpu());
        // Plan should reflect this
        assert!(pipeline.plan().embeddings.is_cpu());
    }

    #[test]
    fn test_builder_quantization_propagation() {
        let hidden = 32;
        let vocab = 100;
        let (_dir, weights) = create_dummy_weights(vec![
            (
                "token_embd.weight",
                vec![1.0; vocab * hidden],
                vec![vocab, hidden],
            ),
            (
                "output.weight",
                vec![1.0; vocab * hidden],
                vec![vocab, hidden],
            ),
        ]);

        let config = Arc::new(MockConfig {
            hidden,
            vocab,
            tied: false,
        });

        let load_config = ModelLoadConfig {
            quantize_lm_head: Some(DType::Q8_0),
            ..Default::default()
        };
        let (cpu, gpu) = get_mock_backends();
        let pipeline = DecoderPipelineBuilder::new(&weights, config)
            .with_backends(cpu, gpu)
            .with_load_config(load_config)
            .build()
            .unwrap();

        let head = pipeline.lm_head().cpu_weights.as_ref().unwrap();
        assert_eq!(head.dtype(), DType::Q8_0);
    }
}
