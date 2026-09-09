//! Core model traits and data structures for Kjarni.

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::WgpuContext;
use crate::activations::Activation;
pub use crate::cache::Cache;
use crate::cpu::encoder::traits::ClassificationMode;
use crate::models::base::RopeScalingConfig;
use anyhow::Result;
use ndarray::Array3;
use std::any::Any;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use std::sync::Arc;

/// Compute backend for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Wgpu,
}

impl Device {
    /// Returns `true` if this is the CPU backend.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns `true` if this is the WebGPU backend.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Wgpu)
    }
}

/// A handle to a loaded model instance.
pub trait InferenceModel: Send + Sync {
    /// Returns the device this model is running on.
    fn device(&self) -> Device;

    /// Returns the GPU context, if running on WebGPU.
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }

    /// Downcasts to a concrete type.
    fn as_any(&self) -> &dyn Any;
}

/// Layer normalization strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizationStrategy {
    LayerNorm,
    RMSNorm,
}

/// Core hyperparameters and configuration for a transformer model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMetadata {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub norm_eps: f32,
    pub activation: Activation,
    pub rope_theta: Option<f32>,
    pub rope_scaling: Option<RopeScalingConfig>,
    /// Whether to multiply embeddings by `sqrt(hidden_size)`.
    pub scale_embeddings: bool,
    pub extra_pos_embeddings: usize,
    pub transpose_ffn_weights: bool,
    pub transpose_attention_weights: bool,
    pub is_prenorm: bool,
    /// Whether to apply layer norm after embedding lookup.
    pub normalize_embedding: bool,
    pub normalization_strategy: NormalizationStrategy,
    pub no_scale_qk: bool,
    pub decoder_layers: Option<usize>,
    pub intermediate_size: usize,
    pub problem_type: Option<String>,
}

/// Weight tensor names for an attention block.
#[derive(Debug, Clone)]
pub struct AttentionLayout {
    pub q_weight: String,
    pub q_bias: Option<String>,
    pub k_weight: String,
    pub k_bias: Option<String>,
    pub v_weight: String,
    pub v_bias: Option<String>,
    pub o_weight: String,
    pub o_bias: Option<String>,
    pub norm_weight: String,
    pub norm_bias: Option<String>,
}

/// Weight tensor names for a feed-forward block.
#[derive(Debug, Clone)]
pub struct FeedForwardLayout {
    pub up_weight: String,
    pub up_bias: Option<String>,
    pub down_weight: String,
    pub down_bias: Option<String>,
    /// For gated variants (SwiGLU, GEGLU).
    pub gate_weight: Option<String>,
    pub gate_bias: Option<String>,
    pub norm_weight: String,
    pub norm_bias: Option<String>,
}

/// Weight tensor names for an encoder layer.
#[derive(Debug, Clone)]
pub struct EncoderLayerLayout {
    pub self_attn: AttentionLayout,
    pub ffn: FeedForwardLayout,
}

/// Weight tensor names for a decoder layer.
#[derive(Debug, Clone)]
pub struct DecoderLayerLayout {
    pub self_attn: AttentionLayout,
    /// Only present in encoder-decoder models.
    pub cross_attn: Option<AttentionLayout>,
    pub ffn: FeedForwardLayout,
}

/// Weight tensor names for an encoder block.
#[derive(Debug, Clone)]
pub struct EncoderLayout {
    pub position_embedding: Option<String>,
    pub token_type_embedding: Option<String>,
    pub embedding_norm_weight: Option<String>,
    pub embedding_norm_bias: Option<String>,
    pub final_norm_weight: Option<String>,
    pub final_norm_bias: Option<String>,
    pub layer: EncoderLayerLayout,
}

/// Weight tensor names for a decoder block.
#[derive(Debug, Clone)]
pub struct DecoderLayout {
    pub position_embedding: Option<String>,
    pub token_type_embedding: Option<String>,
    pub embedding_norm_weight: Option<String>,
    pub embedding_norm_bias: Option<String>,
    pub final_norm_weight: Option<String>,
    pub final_norm_bias: Option<String>,
    pub layer: DecoderLayerLayout,
}

///  tensor layout for any transformer
///
/// - Decoder-only (Llama): `encoder` is `None`
/// - Encoder-only (BERT): `decoder` is `None`
/// - Encoder-decoder (BART): both are `Some`
#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub token_embedding: String,
    pub lm_head: String,
    pub encoder: Option<EncoderLayout>,
    pub decoder: Option<DecoderLayout>,
}

/// Configuration trait implemented by all model configs.
pub trait ModelConfig: Send + Sync {
    fn metadata(&self) -> ModelMetadata;
    fn layout(&self) -> ModelLayout;
    fn model_type(&self) -> &str;
    fn as_any(&self) -> &dyn std::any::Any;

    fn vocab_size(&self) -> usize {
        self.metadata().vocab_size
    }

    fn num_heads(&self) -> usize {
        self.metadata().num_attention_heads
    }

    fn context_size(&self) -> usize {
        self.metadata().max_seq_len
    }

    fn hidden_size(&self) -> usize {
        self.metadata().hidden_size
    }

    fn num_attention_heads(&self) -> usize {
        self.metadata().num_attention_heads
    }

    fn num_layers(&self) -> usize {
        self.metadata().num_layers
    }

    fn layer_norm_eps(&self) -> f32 {
        self.metadata().norm_eps
    }

    fn activation(&self) -> Activation {
        self.metadata().activation
    }

    fn intermediate_size(&self) -> usize {
        0
    }

    fn eos_token_id(&self) -> Option<u32> {
        None
    }

    /// Returns ordered label names for classification models.
    fn id2label(&self) -> Option<&[String]> {
        None
    }

    /// Returns the number of classification labels.
    fn num_labels(&self) -> Option<usize> {
        self.id2label().map(|l| l.len())
    }

    fn default_classification_mode(&self) -> Option<ClassificationMode> {
        ClassificationMode::SingleLabel.into()
    }
    fn is_multi_label(&self) -> bool {
        self.metadata().problem_type.as_deref() == Some("multi_label_classification")
    }
}

/// Shared functionality for CPU transformer blocks.
pub trait CpuTransformerCore: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Applies post-embedding normalization. Default is passthrough.
    fn embed_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        Ok(hidden_states.clone())
    }

    /// Applies final layer normalization.
    fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;

    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_attention_heads(&self) -> usize;

    fn num_kv_heads(&self) -> usize {
        self.num_attention_heads()
    }

    fn head_dim(&self) -> usize {
        self.hidden_size() / self.num_attention_heads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_methods() {
        let cpu = Device::Cpu;
        let gpu = Device::Wgpu;

        assert!(cpu.is_cpu());
        assert!(!cpu.is_gpu());
        assert!(gpu.is_gpu());
        assert!(!gpu.is_cpu());
    }

    struct DummyModel {
        device: Device,
    }

    impl InferenceModel for DummyModel {
        fn device(&self) -> Device {
            self.device
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_inference_model_trait() {
        let model = DummyModel {
            device: Device::Cpu,
        };
        let model_ref: &dyn InferenceModel = &model;

        assert_eq!(model_ref.device(), Device::Cpu);

        let downcasted = model_ref.as_any().downcast_ref::<DummyModel>().unwrap();
        assert_eq!(downcasted.device, Device::Cpu);
    }

    #[test]
    fn test_model_metadata_fields() {
        let meta = ModelMetadata {
            decoder_layers: None,
            hidden_size: 128,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 4,
            head_dim: 32,
            vocab_size: 1000,
            intermediate_size: 0,
            max_seq_len: 512,
            norm_eps: 1e-5,
            activation: Activation::Gelu,
            rope_theta: Some(1000.0),
            rope_scaling: None,
            scale_embeddings: true,
            extra_pos_embeddings: 0,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            problem_type: None,
            is_prenorm: true,
            normalize_embedding: false,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
        };

        assert_eq!(meta.hidden_size, 128);
        assert_eq!(meta.num_layers, 2);
        assert_eq!(meta.activation, Activation::Gelu);
        assert!(meta.rope_theta.is_some());
    }

    fn make_attention_layout() -> AttentionLayout {
        AttentionLayout {
            q_weight: "q".to_string(),
            q_bias: None,
            k_weight: "k".to_string(),
            k_bias: None,
            v_weight: "v".to_string(),
            v_bias: None,
            o_weight: "o".to_string(),
            o_bias: None,
            norm_weight: "n".to_string(),
            norm_bias: None,
        }
    }

    fn make_ffn_layout() -> FeedForwardLayout {
        FeedForwardLayout {
            up_weight: "up".to_string(),
            up_bias: None,
            down_weight: "down".to_string(),
            down_bias: None,
            gate_weight: None,
            gate_bias: None,
            norm_weight: "norm".to_string(),
            norm_bias: None,
        }
    }

    #[test]
    fn test_attention_layout() {
        let attn = AttentionLayout {
            q_weight: "q_w".to_string(),
            q_bias: Some("q_b".to_string()),
            k_weight: "k_w".to_string(),
            k_bias: None,
            v_weight: "v_w".to_string(),
            v_bias: None,
            o_weight: "o_w".to_string(),
            o_bias: None,
            norm_weight: "norm_w".to_string(),
            norm_bias: Some("norm_b".to_string()),
        };

        assert_eq!(attn.q_weight, "q_w");
        assert!(attn.k_bias.is_none());
    }

    #[test]
    fn test_feedforward_layout() {
        let ffn = FeedForwardLayout {
            up_weight: "up_w".to_string(),
            up_bias: None,
            down_weight: "down_w".to_string(),
            down_bias: Some("down_b".to_string()),
            gate_weight: None,
            gate_bias: None,
            norm_weight: "norm_w".to_string(),
            norm_bias: None,
        };

        assert_eq!(ffn.down_weight, "down_w");
        assert!(ffn.gate_weight.is_none());
    }

    #[test]
    fn test_encoder_and_decoder_layer_layouts() {
        let attn = make_attention_layout();
        let ffn = make_ffn_layout();

        let encoder_layer = EncoderLayerLayout {
            self_attn: attn.clone(),
            ffn: ffn.clone(),
        };
        let decoder_layer = DecoderLayerLayout {
            self_attn: attn.clone(),
            cross_attn: Some(attn.clone()),
            ffn: ffn.clone(),
        };

        assert_eq!(encoder_layer.ffn.down_weight, "down");
        assert!(decoder_layer.cross_attn.is_some());
    }

    #[test]
    fn test_encoder_and_decoder_layouts() {
        let attn = make_attention_layout();
        let ffn = make_ffn_layout();

        let encoder_layout = EncoderLayout {
            position_embedding: Some("pos".to_string()),
            token_type_embedding: None,
            embedding_norm_weight: Some("emb_norm".to_string()),
            embedding_norm_bias: None,
            final_norm_weight: None,
            final_norm_bias: None,
            layer: EncoderLayerLayout {
                self_attn: attn.clone(),
                ffn: ffn.clone(),
            },
        };

        let decoder_layout = DecoderLayout {
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_weight: None,
            embedding_norm_bias: None,
            final_norm_weight: Some("final_norm".to_string()),
            final_norm_bias: None,
            layer: DecoderLayerLayout {
                self_attn: attn.clone(),
                cross_attn: Some(attn.clone()),
                ffn: ffn.clone(),
            },
        };

        assert_eq!(encoder_layout.layer.ffn.up_weight, "up");
        assert_eq!(decoder_layout.layer.self_attn.q_weight, "q");
    }

    #[test]
    fn test_model_layout() {
        let layout = ModelLayout {
            token_embedding: "tok_emb".to_string(),
            lm_head: "lm_head".to_string(),
            encoder: None,
            decoder: None,
        };

        assert_eq!(layout.token_embedding, "tok_emb");
        assert!(layout.encoder.is_none());
        assert!(layout.decoder.is_none());
    }
}
