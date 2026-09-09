//! Builder for EncoderPipeline.

use anyhow::{Result, anyhow};
use std::sync::Arc;

use crate::WgpuContext;
use crate::execution::ExecutionPlan;
use crate::models::base::ModelLoadConfig;
use crate::traits::{Device, ModelConfig};
use crate::weights::ModelWeights;
use crate::{EmbeddingConfig, LoadedEmbeddings};
use crate::{
    cpu::encoder::{
        classifier::CpuSequenceClassificationHead,
        config::PoolingStrategy,
        traits::{CpuEncoder, GpuEncoder},
    },
    pipeline::encoder::pipeline::{EncoderPipeline, EncoderPipelineConfig},
};

/// Builder for constructing an EncoderPipeline.
pub struct EncoderPipelineBuilder<'a> {
    weights: &'a ModelWeights,
    config: Arc<dyn ModelConfig>,
    load_config: ModelLoadConfig,
    context: Option<Arc<WgpuContext>>,

    // Backends
    cpu_encoder: Option<Box<dyn CpuEncoder>>,
    gpu_encoder: Option<Box<dyn GpuEncoder>>,

    // Optional head
    cpu_head: Option<CpuSequenceClassificationHead>,

    // Pooling strategy
    pooling_strategy: PoolingStrategy,
}

impl<'a> EncoderPipelineBuilder<'a> {
    pub fn new(weights: &'a ModelWeights, config: Arc<dyn ModelConfig>) -> Self {
        Self {
            weights,
            config,
            load_config: ModelLoadConfig::default(),
            context: None,
            cpu_encoder: None,
            gpu_encoder: None,
            cpu_head: None,
            pooling_strategy: PoolingStrategy::Mean, // Default for embeddings
        }
    }

    pub fn with_load_config(mut self, cfg: ModelLoadConfig) -> Self {
        self.load_config = cfg;
        self
    }

    pub fn with_context(mut self, ctx: Option<Arc<WgpuContext>>) -> Self {
        self.context = ctx;
        self
    }

    pub fn with_backends(
        mut self,
        cpu: Option<Box<dyn CpuEncoder>>,
        gpu: Option<Box<dyn GpuEncoder>>,
    ) -> Self {
        self.cpu_encoder = cpu;
        self.gpu_encoder = gpu;
        self
    }

    pub fn with_head(mut self, head: Option<CpuSequenceClassificationHead>) -> Self {
        self.cpu_head = head;
        self
    }

    pub fn with_pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.pooling_strategy = strategy;
        self
    }

    pub fn build(self) -> Result<EncoderPipeline> {
        let meta = self.config.metadata();
        let layout = self.config.layout();
        let ctx = self.context.as_ref();
        let target_dtype = self.load_config.target_dtype;

        // Determine execution plan
        let primary_device = if ctx.is_some() {
            Device::Wgpu
        } else {
            Device::Cpu
        };
        let plan = ExecutionPlan::from_load_config(primary_device, &self.load_config);

        // Get encoder layout
        let enc_layout = layout
            .encoder
            .as_ref()
            .ok_or_else(|| anyhow!("Pipeline requires an EncoderLayout in ModelLayout"))?;

        // Load embeddings
        let mut emb_builder = EmbeddingConfig::builder(&layout.token_embedding, meta.hidden_size)
            .position_offset(meta.extra_pos_embeddings);

        if let Some(pos) = &enc_layout.position_embedding {
            emb_builder = emb_builder.position_embedding(pos);
        }
        if let Some(tok) = &enc_layout.token_type_embedding {
            emb_builder = emb_builder.type_embedding(tok);
        }

        let emb_load_cpu = plan.embeddings == Device::Cpu;
        let emb_load_gpu = plan.embeddings == Device::Wgpu;

        // The wasm constructor takes no context and no device flags, since there is
        // only ever the CPU path there.
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
        let embeddings = LoadedEmbeddings::new(
            ctx,
            self.weights,
            emb_builder.build(),
            emb_load_cpu,
            emb_load_gpu,
            target_dtype,
        )?;
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
        let embeddings = {
            let _ = (emb_load_cpu, emb_load_gpu);
            LoadedEmbeddings::new(self.weights, emb_builder.build(), target_dtype)?
        };

        //  Build pipeline config
        let pipeline_config = EncoderPipelineConfig {
            num_layers: meta.num_layers,
            hidden_size: meta.hidden_size,
            vocab_size: meta.vocab_size,
            max_seq_length: meta.max_seq_len,
            pooling_strategy: self.pooling_strategy,
            has_head: self.cpu_head.is_some(),
            num_labels: self.cpu_head.as_ref().map(|h| h.num_classes()),
        };

        // Build pipeline
        EncoderPipeline::new(
            embeddings,
            self.cpu_encoder,
            self.gpu_encoder,
            self.cpu_head,
            plan,
            self.context,
            pipeline_config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::cpu::encoder::config::PoolingStrategy;
    use crate::models::base::ModelLoadConfig;
    use crate::traits::{
        AttentionLayout, DecoderLayerLayout, DecoderLayout, EncoderLayerLayout, EncoderLayout,
        FeedForwardLayout, ModelConfig, ModelLayout, ModelMetadata, NormalizationStrategy,
    };
    struct MockModelConfig {
        metadata: ModelMetadata,
        layout: ModelLayout,
    }

    impl MockModelConfig {
        fn new() -> Self {
            Self {
                metadata: ModelMetadata {
                    hidden_size: 768,
                    num_layers: 12,
                    num_attention_heads: 12,
                    num_kv_heads: 12,
                    head_dim: 64,
                    vocab_size: 30522,
                    max_seq_len: 512,
                    norm_eps: 1e-12,
                    activation: Activation::Gelu,
                    rope_theta: None,
                    rope_scaling: None,
                    scale_embeddings: false,
                    extra_pos_embeddings: 0,
                    transpose_ffn_weights: false,
                    transpose_attention_weights: false,
                    problem_type: None,
                    is_prenorm: false,
                    normalize_embedding: true,
                    normalization_strategy: NormalizationStrategy::LayerNorm,
                    no_scale_qk: false,
                    decoder_layers: None,
                    intermediate_size: 3072,
                },
                layout: ModelLayout {
                    token_embedding: "embeddings.word_embeddings.weight".to_string(),
                    lm_head: "cls.predictions.decoder.weight".to_string(),
                    encoder: None,
                    decoder: None,
                },
            }
        }

        fn with_encoder_layout(mut self) -> Self {
            self.layout.encoder = Some(EncoderLayout {
                position_embedding: Some("embeddings.position_embeddings.weight".to_string()),
                token_type_embedding: Some("embeddings.token_type_embeddings.weight".to_string()),
                embedding_norm_weight: Some("embeddings.LayerNorm.weight".to_string()),
                embedding_norm_bias: Some("embeddings.LayerNorm.bias".to_string()),
                final_norm_weight: None,
                final_norm_bias: None,
                layer: EncoderLayerLayout {
                    self_attn: AttentionLayout {
                        q_weight: "attention.self.query.weight".to_string(),
                        q_bias: Some("attention.self.query.bias".to_string()),
                        k_weight: "attention.self.key.weight".to_string(),
                        k_bias: Some("attention.self.key.bias".to_string()),
                        v_weight: "attention.self.value.weight".to_string(),
                        v_bias: Some("attention.self.value.bias".to_string()),
                        o_weight: "attention.output.dense.weight".to_string(),
                        o_bias: Some("attention.output.dense.bias".to_string()),
                        norm_weight: "attention.output.LayerNorm.weight".to_string(),
                        norm_bias: Some("attention.output.LayerNorm.bias".to_string()),
                    },
                    ffn: FeedForwardLayout {
                        up_weight: "intermediate.dense.weight".to_string(),
                        up_bias: Some("intermediate.dense.bias".to_string()),
                        down_weight: "output.dense.weight".to_string(),
                        down_bias: Some("output.dense.bias".to_string()),
                        gate_weight: None,
                        gate_bias: None,
                        norm_weight: "output.LayerNorm.weight".to_string(),
                        norm_bias: Some("output.LayerNorm.bias".to_string()),
                    },
                },
            });
            self
        }

        fn with_decoder_layout(mut self) -> Self {
            self.layout.decoder = Some(DecoderLayout {
                position_embedding: Some("decoder.embed_positions.weight".to_string()),
                token_type_embedding: None,
                embedding_norm_weight: Some("decoder.layernorm_embedding.weight".to_string()),
                embedding_norm_bias: Some("decoder.layernorm_embedding.bias".to_string()),
                final_norm_weight: None,
                final_norm_bias: None,
                layer: DecoderLayerLayout {
                    self_attn: AttentionLayout {
                        q_weight: "self_attn.q_proj.weight".to_string(),
                        q_bias: Some("self_attn.q_proj.bias".to_string()),
                        k_weight: "self_attn.k_proj.weight".to_string(),
                        k_bias: Some("self_attn.k_proj.bias".to_string()),
                        v_weight: "self_attn.v_proj.weight".to_string(),
                        v_bias: Some("self_attn.v_proj.bias".to_string()),
                        o_weight: "self_attn.out_proj.weight".to_string(),
                        o_bias: Some("self_attn.out_proj.bias".to_string()),
                        norm_weight: "self_attn_layer_norm.weight".to_string(),
                        norm_bias: Some("self_attn_layer_norm.bias".to_string()),
                    },
                    cross_attn: Some(AttentionLayout {
                        q_weight: "encoder_attn.q_proj.weight".to_string(),
                        q_bias: Some("encoder_attn.q_proj.bias".to_string()),
                        k_weight: "encoder_attn.k_proj.weight".to_string(),
                        k_bias: Some("encoder_attn.k_proj.bias".to_string()),
                        v_weight: "encoder_attn.v_proj.weight".to_string(),
                        v_bias: Some("encoder_attn.v_proj.bias".to_string()),
                        o_weight: "encoder_attn.out_proj.weight".to_string(),
                        o_bias: Some("encoder_attn.out_proj.bias".to_string()),
                        norm_weight: "encoder_attn_layer_norm.weight".to_string(),
                        norm_bias: Some("encoder_attn_layer_norm.bias".to_string()),
                    }),
                    ffn: FeedForwardLayout {
                        up_weight: "fc1.weight".to_string(),
                        up_bias: Some("fc1.bias".to_string()),
                        down_weight: "fc2.weight".to_string(),
                        down_bias: Some("fc2.bias".to_string()),
                        gate_weight: None,
                        gate_bias: None,
                        norm_weight: "final_layer_norm.weight".to_string(),
                        norm_bias: Some("final_layer_norm.bias".to_string()),
                    },
                },
            });
            self
        }
    }

    impl ModelConfig for MockModelConfig {
        fn metadata(&self) -> ModelMetadata {
            self.metadata.clone()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn model_type(&self) -> &str {
            "Mock"
        }

        fn layout(&self) -> ModelLayout {
            self.layout.clone()
        }
    }
    #[test]
    fn test_pooling_strategy_variants() {
        let mean = PoolingStrategy::Mean;
        let cls = PoolingStrategy::Cls;
        let last = PoolingStrategy::LastToken;

        assert!(matches!(mean, PoolingStrategy::Mean));
        assert!(matches!(cls, PoolingStrategy::Cls));
        assert!(matches!(last, PoolingStrategy::LastToken));
    }
    #[test]
    fn test_mock_model_config_metadata() {
        let config = MockModelConfig::new();
        let meta = config.metadata();

        assert_eq!(meta.hidden_size, 768);
        assert_eq!(meta.num_layers, 12);
        assert_eq!(meta.num_attention_heads, 12);
        assert_eq!(meta.num_kv_heads, 12);
        assert_eq!(meta.head_dim, 64);
        assert_eq!(meta.vocab_size, 30522);
        assert_eq!(meta.max_seq_len, 512);
        assert_eq!(meta.intermediate_size, 3072);
    }

    #[test]
    fn test_mock_model_config_layout_default() {
        let config = MockModelConfig::new();
        let layout = config.layout();

        assert!(layout.encoder.is_none());
        assert!(layout.decoder.is_none());
        assert!(!layout.token_embedding.is_empty());
        assert!(!layout.lm_head.is_empty());
    }

    #[test]
    fn test_mock_model_config_with_encoder_layout() {
        let config = MockModelConfig::new().with_encoder_layout();
        let layout = config.layout();

        assert!(layout.encoder.is_some());
        let enc = layout.encoder.as_ref().unwrap();
        assert!(enc.position_embedding.is_some());
        assert!(enc.token_type_embedding.is_some());
        assert!(enc.embedding_norm_weight.is_some());
    }

    #[test]
    fn test_mock_model_config_with_decoder_layout() {
        let config = MockModelConfig::new().with_decoder_layout();
        let layout = config.layout();

        assert!(layout.decoder.is_some());
        let dec = layout.decoder.as_ref().unwrap();
        assert!(dec.position_embedding.is_some());
        assert!(dec.layer.cross_attn.is_some());
    }

    #[test]
    fn test_encoder_pipeline_config_creation() {
        let config = EncoderPipelineConfig {
            num_layers: 12,
            hidden_size: 768,
            vocab_size: 30522,
            max_seq_length: 512,
            pooling_strategy: PoolingStrategy::Mean,
            has_head: false,
            num_labels: None,
        };

        assert_eq!(config.num_layers, 12);
        assert_eq!(config.hidden_size, 768);
        assert!(!config.has_head);
    }

    #[test]
    fn test_encoder_pipeline_config_with_head() {
        let config = EncoderPipelineConfig {
            num_layers: 6,
            hidden_size: 384,
            vocab_size: 30522,
            max_seq_length: 256,
            pooling_strategy: PoolingStrategy::Cls,
            has_head: true,
            num_labels: Some(2),
        };

        assert!(config.has_head);
        assert_eq!(config.num_labels, Some(2));
    }

    #[test]
    fn test_execution_plan_from_load_config_cpu() {
        let load_config = ModelLoadConfig::default();
        let plan = ExecutionPlan::from_load_config(Device::Cpu, &load_config);

        assert_eq!(plan.embeddings, Device::Cpu);
    }

    #[test]
    fn test_execution_plan_from_load_config_gpu() {
        let load_config = ModelLoadConfig::default();
        let plan = ExecutionPlan::from_load_config(Device::Wgpu, &load_config);

        assert_eq!(plan.embeddings, Device::Wgpu);
    }

    #[test]
    fn test_device_selection_with_context() {
        let has_context = true;
        let primary_device = if has_context {
            Device::Wgpu
        } else {
            Device::Cpu
        };
        assert_eq!(primary_device, Device::Wgpu);
    }

    #[test]
    fn test_device_selection_without_context() {
        let has_context = false;
        let primary_device = if has_context {
            Device::Wgpu
        } else {
            Device::Cpu
        };
        assert_eq!(primary_device, Device::Cpu);
    }
    #[test]
    fn test_model_metadata_bert_style() {
        let meta = ModelMetadata {
            hidden_size: 768,
            num_layers: 12,
            num_attention_heads: 12,
            num_kv_heads: 12,
            head_dim: 64,
            vocab_size: 30522,
            max_seq_len: 512,
            norm_eps: 1e-12,
            activation: Activation::Gelu,
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: false,
            extra_pos_embeddings: 0,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            is_prenorm: false,
            normalize_embedding: true,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
            decoder_layers: None,
            problem_type: None,
            intermediate_size: 3072,
        };

        assert!(matches!(
            meta.normalization_strategy,
            NormalizationStrategy::LayerNorm
        ));
        assert!(meta.rope_theta.is_none());
        assert!(!meta.is_prenorm);
        assert!(meta.normalize_embedding);
    }

    #[test]
    fn test_model_metadata_llama_style() {
        let meta = ModelMetadata {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8, // GQA
            problem_type: None,
            head_dim: 128,
            vocab_size: 32000,
            max_seq_len: 4096,
            norm_eps: 1e-6,
            activation: Activation::SilU,
            rope_theta: Some(10000.0),
            rope_scaling: None,
            scale_embeddings: false,
            extra_pos_embeddings: 0,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            is_prenorm: true,
            normalize_embedding: false,
            normalization_strategy: NormalizationStrategy::RMSNorm,
            no_scale_qk: false,
            decoder_layers: None,
            intermediate_size: 11008,
        };

        assert!(matches!(
            meta.normalization_strategy,
            NormalizationStrategy::RMSNorm
        ));
        assert!(meta.rope_theta.is_some());
        assert!(meta.is_prenorm);
        assert!(!meta.normalize_embedding);
        assert_eq!(meta.num_kv_heads, 8); // GQA
    }

    #[test]
    fn test_model_metadata_bart_style() {
        let meta = ModelMetadata {
            hidden_size: 1024,
            num_layers: 12,
            num_attention_heads: 16,
            num_kv_heads: 16,
            head_dim: 64,
            vocab_size: 50265,
            max_seq_len: 1024,
            norm_eps: 1e-5,
            activation: Activation::Gelu,
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: true,  // BART scales embeddings
            extra_pos_embeddings: 2, // BART has offset
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            is_prenorm: false,
            normalize_embedding: true,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
            problem_type: None,
            decoder_layers: Some(12),
            intermediate_size: 4096,
        };

        assert!(meta.scale_embeddings);
        assert_eq!(meta.extra_pos_embeddings, 2);
        assert_eq!(meta.decoder_layers, Some(12));
    }

    #[test]
    fn test_attention_layout_with_biases() {
        let layout = AttentionLayout {
            q_weight: "q.weight".to_string(),
            q_bias: Some("q.bias".to_string()),
            k_weight: "k.weight".to_string(),
            k_bias: Some("k.bias".to_string()),
            v_weight: "v.weight".to_string(),
            v_bias: Some("v.bias".to_string()),
            o_weight: "o.weight".to_string(),
            o_bias: Some("o.bias".to_string()),
            norm_weight: "norm.weight".to_string(),
            norm_bias: Some("norm.bias".to_string()),
        };

        assert!(layout.q_bias.is_some());
        assert!(layout.k_bias.is_some());
        assert!(layout.v_bias.is_some());
        assert!(layout.o_bias.is_some());
        assert!(layout.norm_bias.is_some());
    }

    #[test]
    fn test_attention_layout_without_biases() {
        let layout = AttentionLayout {
            q_weight: "q.weight".to_string(),
            q_bias: None,
            k_weight: "k.weight".to_string(),
            k_bias: None,
            v_weight: "v.weight".to_string(),
            v_bias: None,
            o_weight: "o.weight".to_string(),
            o_bias: None,
            norm_weight: "norm.weight".to_string(),
            norm_bias: None,
        };

        assert!(layout.q_bias.is_none());
        assert!(layout.k_bias.is_none());
        assert!(layout.v_bias.is_none());
        assert!(layout.o_bias.is_none());
        assert!(layout.norm_bias.is_none());
    }

    #[test]
    fn test_feedforward_layout_standard() {
        let layout = FeedForwardLayout {
            up_weight: "up.weight".to_string(),
            up_bias: Some("up.bias".to_string()),
            down_weight: "down.weight".to_string(),
            down_bias: Some("down.bias".to_string()),
            gate_weight: None,
            gate_bias: None,
            norm_weight: "norm.weight".to_string(),
            norm_bias: Some("norm.bias".to_string()),
        };

        assert!(layout.gate_weight.is_none());
        assert!(layout.gate_bias.is_none());
    }

    #[test]
    fn test_feedforward_layout_gated() {
        let layout = FeedForwardLayout {
            up_weight: "up.weight".to_string(),
            up_bias: None,
            down_weight: "down.weight".to_string(),
            down_bias: None,
            gate_weight: Some("gate.weight".to_string()),
            gate_bias: None,
            norm_weight: "norm.weight".to_string(),
            norm_bias: None,
        };

        assert!(layout.gate_weight.is_some());
    }

    #[test]
    fn test_encoder_layout_full() {
        let layout = EncoderLayout {
            position_embedding: Some("pos.weight".to_string()),
            token_type_embedding: Some("type.weight".to_string()),
            embedding_norm_weight: Some("embed_norm.weight".to_string()),
            embedding_norm_bias: Some("embed_norm.bias".to_string()),
            final_norm_weight: Some("final_norm.weight".to_string()),
            final_norm_bias: Some("final_norm.bias".to_string()),
            layer: EncoderLayerLayout {
                self_attn: AttentionLayout {
                    q_weight: "q.weight".to_string(),
                    q_bias: None,
                    k_weight: "k.weight".to_string(),
                    k_bias: None,
                    v_weight: "v.weight".to_string(),
                    v_bias: None,
                    o_weight: "o.weight".to_string(),
                    o_bias: None,
                    norm_weight: "attn_norm.weight".to_string(),
                    norm_bias: None,
                },
                ffn: FeedForwardLayout {
                    up_weight: "up.weight".to_string(),
                    up_bias: None,
                    down_weight: "down.weight".to_string(),
                    down_bias: None,
                    gate_weight: None,
                    gate_bias: None,
                    norm_weight: "ffn_norm.weight".to_string(),
                    norm_bias: None,
                },
            },
        };

        assert!(layout.position_embedding.is_some());
        assert!(layout.token_type_embedding.is_some());
        assert!(layout.final_norm_weight.is_some());
    }

    #[test]
    fn test_encoder_layout_minimal() {
        let layout = EncoderLayout {
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_weight: None,
            embedding_norm_bias: None,
            final_norm_weight: None,
            final_norm_bias: None,
            layer: EncoderLayerLayout {
                self_attn: AttentionLayout {
                    q_weight: "q.weight".to_string(),
                    q_bias: None,
                    k_weight: "k.weight".to_string(),
                    k_bias: None,
                    v_weight: "v.weight".to_string(),
                    v_bias: None,
                    o_weight: "o.weight".to_string(),
                    o_bias: None,
                    norm_weight: "attn_norm.weight".to_string(),
                    norm_bias: None,
                },
                ffn: FeedForwardLayout {
                    up_weight: "up.weight".to_string(),
                    up_bias: None,
                    down_weight: "down.weight".to_string(),
                    down_bias: None,
                    gate_weight: None,
                    gate_bias: None,
                    norm_weight: "ffn_norm.weight".to_string(),
                    norm_bias: None,
                },
            },
        };

        assert!(layout.position_embedding.is_none());
        assert!(layout.token_type_embedding.is_none());
    }

    #[test]
    fn test_decoder_layout_with_cross_attention() {
        let config = MockModelConfig::new().with_decoder_layout();
        let dec = config.layout.decoder.as_ref().unwrap();

        assert!(dec.layer.cross_attn.is_some());
    }

    #[test]
    fn test_decoder_layout_without_cross_attention() {
        let layout = DecoderLayout {
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_weight: None,
            embedding_norm_bias: None,
            final_norm_weight: Some("final_norm.weight".to_string()),
            final_norm_bias: None,
            layer: DecoderLayerLayout {
                self_attn: AttentionLayout {
                    q_weight: "q.weight".to_string(),
                    q_bias: None,
                    k_weight: "k.weight".to_string(),
                    k_bias: None,
                    v_weight: "v.weight".to_string(),
                    v_bias: None,
                    o_weight: "o.weight".to_string(),
                    o_bias: None,
                    norm_weight: "attn_norm.weight".to_string(),
                    norm_bias: None,
                },
                cross_attn: None, // Decoder-only model
                ffn: FeedForwardLayout {
                    up_weight: "up.weight".to_string(),
                    up_bias: None,
                    down_weight: "down.weight".to_string(),
                    down_bias: None,
                    gate_weight: Some("gate.weight".to_string()),
                    gate_bias: None,
                    norm_weight: "ffn_norm.weight".to_string(),
                    norm_bias: None,
                },
            },
        };

        assert!(layout.layer.cross_attn.is_none());
        assert!(layout.layer.ffn.gate_weight.is_some()); // SwiGLU
    }

    #[test]
    fn test_model_layout_encoder_only() {
        let config = MockModelConfig::new().with_encoder_layout();
        let layout = config.layout();

        assert!(layout.encoder.is_some());
        assert!(layout.decoder.is_none());
    }

    #[test]
    fn test_model_layout_decoder_only() {
        let config = MockModelConfig::new().with_decoder_layout();
        let layout = config.layout();

        assert!(layout.encoder.is_none());
        assert!(layout.decoder.is_some());
    }

    #[test]
    fn test_model_layout_encoder_decoder() {
        let config = MockModelConfig::new()
            .with_encoder_layout()
            .with_decoder_layout();
        let layout = config.layout();

        assert!(layout.encoder.is_some());
        assert!(layout.decoder.is_some());
    }

    #[test]
    fn test_activation_variants() {
        let gelu = Activation::Gelu;
        let silu = Activation::SilU;
        let relu = Activation::Relu;

        assert!(matches!(gelu, Activation::Gelu));
        assert!(matches!(silu, Activation::SilU));
        assert!(matches!(relu, Activation::Relu));
    }

    #[test]
    fn test_normalization_strategy_variants() {
        let layer_norm = NormalizationStrategy::LayerNorm;
        let rms_norm = NormalizationStrategy::RMSNorm;

        assert!(matches!(layer_norm, NormalizationStrategy::LayerNorm));
        assert!(matches!(rms_norm, NormalizationStrategy::RMSNorm));
    }
}
