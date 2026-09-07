//! CLIP vision configuration.
//!
//! Only the vision tower is described here. CLIP's `config.json` nests
//! `vision_config` and `text_config` under one root, and the two are independent
//! transformers that meet only at the projection into a shared space.

use anyhow::{Context, Result};
use kjarni_transformers::activations::Activation;
use kjarni_transformers::traits::{
    AttentionLayout, EncoderLayerLayout, EncoderLayout, FeedForwardLayout, ModelConfig,
    ModelLayout, ModelMetadata, NormalizationStrategy,
};
use serde::Deserialize;

/// The `vision_config` object inside CLIP's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ClipVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: Option<usize>,

    /// `quick_gelu` on OpenAI's checkpoints, `gelu` on LAION's. The two differ
    /// enough to shift embeddings, so it is read rather than assumed.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
}

fn default_hidden_act() -> String {
    "quick_gelu".to_string()
}

fn default_layer_norm_eps() -> f32 {
    1e-5
}

/// The `text_config` object inside CLIP's `config.json`.
///
/// Narrower than the vision tower on B/32: 512 wide against 768, 8 heads against
/// 12. The towers are independent and meet only at the shared projection.
#[derive(Debug, Clone, Deserialize)]
pub struct ClipTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,

    /// 77 for every CLIP released so far, and a hard cap rather than a default:
    /// the position table has exactly this many rows.
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_text_act")]
    pub hidden_act: String,

    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
}

fn default_max_positions() -> usize {
    77
}

fn default_text_act() -> String {
    "quick_gelu".to_string()
}

/// The parts of the CLIP root config this loader needs.
#[derive(Debug, Clone, Deserialize)]
pub struct ClipConfig {
    pub vision_config: ClipVisionConfig,

    /// Absent on vision-only exports, which is why this is optional rather than
    /// required: a plain ViT has no text tower at all.
    #[serde(default)]
    pub text_config: Option<ClipTextConfig>,

    /// Width of the shared image/text space, which is what `visual_projection`
    /// maps into. Smaller than `hidden_size`: 512 against 768 on B/32.
    #[serde(default = "default_projection_dim")]
    pub projection_dim: usize,
}

fn default_projection_dim() -> usize {
    512
}

impl ClipConfig {
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("parsing CLIP config.json")
    }

    /// Patches per side, e.g. 224 / 32 = 7.
    pub fn patches_per_side(&self) -> usize {
        self.vision_config.image_size / self.vision_config.patch_size
    }

    /// Sequence length the tower sees: one position per patch, plus the class
    /// token that carries the pooled representation. 7x7 + 1 = 50 on B/32.
    pub fn num_positions(&self) -> usize {
        self.patches_per_side() * self.patches_per_side() + 1
    }

    pub fn num_channels(&self) -> usize {
        self.vision_config.num_channels.unwrap_or(3)
    }

    /// Flattened patch width: channels * patch * patch, 3072 on B/32. This is the
    /// inner dimension of the single matmul that replaces the strided conv.
    pub fn patch_dim(&self) -> usize {
        self.num_channels() * self.vision_config.patch_size * self.vision_config.patch_size
    }

    pub fn text(&self) -> Result<&ClipTextConfig> {
        self.text_config
            .as_ref()
            .context("this checkpoint has no text_config; it is vision-only")
    }

    pub fn text_activation(&self) -> Result<Activation> {
        parse_activation(&self.text()?.hidden_act)
    }

    pub fn activation(&self) -> Result<Activation> {
        parse_activation(&self.vision_config.hidden_act)
    }
}

impl ModelConfig for ClipConfig {
    fn metadata(&self) -> ModelMetadata {
        let v = &self.vision_config;
        ModelMetadata {
            hidden_size: v.hidden_size,
            num_layers: v.num_hidden_layers,
            num_attention_heads: v.num_attention_heads,
            // No grouped-query attention in CLIP: every head has its own K and V.
            num_kv_heads: v.num_attention_heads,
            head_dim: v.hidden_size / v.num_attention_heads,
            // Positions are fixed by the image geometry, not a vocabulary. A
            // vision tower has no tokens, so this is the patch count instead.
            vocab_size: 0,
            max_seq_len: self.num_positions(),
            norm_eps: v.layer_norm_eps,
            activation: self.activation().unwrap_or(Activation::GeluNew),
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: false,
            extra_pos_embeddings: 0,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            // CLIP norms before each sublayer, unlike BERT which norms after.
            is_prenorm: true,
            // `pre_layrnorm` runs once on the assembled patch sequence.
            normalize_embedding: true,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
            decoder_layers: None,
            intermediate_size: v.intermediate_size,
            problem_type: None,
        }
    }

    fn layout(&self) -> ModelLayout {
        let l = |suffix: &str| format!("vision_model.encoder.layers.{{}}.{suffix}");

        ModelLayout {
            // Patches are projected, not looked up, so the usual embedding table
            // and language head do not exist here.
            token_embedding: "vision_model.embeddings.patch_embedding.weight".to_string(),
            lm_head: String::new(),
            encoder: Some(EncoderLayout {
                position_embedding: Some(
                    "vision_model.embeddings.position_embedding.weight".to_string(),
                ),
                token_type_embedding: None,
                // Spelled `pre_layrnorm` in the checkpoint. The typo is upstream
                // and shipped in every CLIP export, so it has to be matched.
                embedding_norm_weight: Some("vision_model.pre_layrnorm.weight".to_string()),
                embedding_norm_bias: Some("vision_model.pre_layrnorm.bias".to_string()),
                final_norm_weight: Some("vision_model.post_layernorm.weight".to_string()),
                final_norm_bias: Some("vision_model.post_layernorm.bias".to_string()),
                layer: EncoderLayerLayout {
                    self_attn: AttentionLayout {
                        q_weight: l("self_attn.q_proj.weight"),
                        q_bias: Some(l("self_attn.q_proj.bias")),
                        k_weight: l("self_attn.k_proj.weight"),
                        k_bias: Some(l("self_attn.k_proj.bias")),
                        v_weight: l("self_attn.v_proj.weight"),
                        v_bias: Some(l("self_attn.v_proj.bias")),
                        o_weight: l("self_attn.out_proj.weight"),
                        o_bias: Some(l("self_attn.out_proj.bias")),
                        norm_weight: l("layer_norm1.weight"),
                        norm_bias: Some(l("layer_norm1.bias")),
                    },
                    ffn: FeedForwardLayout {
                        up_weight: l("mlp.fc1.weight"),
                        up_bias: Some(l("mlp.fc1.bias")),
                        down_weight: l("mlp.fc2.weight"),
                        down_bias: Some(l("mlp.fc2.bias")),
                        gate_weight: None,
                        gate_bias: None,
                        norm_weight: l("layer_norm2.weight"),
                        norm_bias: Some(l("layer_norm2.bias")),
                    },
                },
            }),
            decoder: None,
        }
    }

    fn model_type(&self) -> &str {
        "clip_vision"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn parse_activation(name: &str) -> Result<Activation> {
    match name {
        "quick_gelu" => Ok(Activation::QuickGelu),
        // HuggingFace maps "gelu" to `GELUActivation`, which is the exact erf
        // form, and "gelu_new" to the tanh approximation. They are close enough
        // to look right and far enough apart to move an embedding, so the two
        // names must not collapse onto one variant here.
        "gelu" => Ok(Activation::Gelu),
        "gelu_new" | "gelu_fast" => Ok(Activation::GeluNew),
        "relu" => Ok(Activation::Relu),
        other => anyhow::bail!("unsupported CLIP activation '{other}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed to the fields the loader reads, so these tests need no checkpoint.
    fn config_json(vision_act: &str, text_act: &str) -> String {
        format!(
            r#"{{
              "projection_dim": 512,
              "vision_config": {{
                "hidden_size": 768, "intermediate_size": 3072,
                "num_hidden_layers": 12, "num_attention_heads": 12,
                "image_size": 224, "patch_size": 32,
                "hidden_act": "{vision_act}", "layer_norm_eps": 1e-5
              }},
              "text_config": {{
                "hidden_size": 512, "intermediate_size": 2048,
                "num_hidden_layers": 12, "num_attention_heads": 8,
                "vocab_size": 49408, "max_position_embeddings": 77,
                "hidden_act": "{text_act}", "layer_norm_eps": 1e-5
              }}
            }}"#
        )
    }

    #[test]
    fn geometry_follows_from_image_and_patch_size() {
        let c = ClipConfig::from_json(&config_json("gelu", "gelu")).unwrap();
        assert_eq!(c.patches_per_side(), 7, "224 / 32");
        // 7x7 patches plus the class token.
        assert_eq!(c.num_positions(), 50);
        // 3 channels x 32 x 32, the width of the flattened patch matmul.
        assert_eq!(c.patch_dim(), 3072);
        assert_eq!(c.num_channels(), 3, "defaults to RGB when unstated");
    }

    /// The distinction that silently degraded the vision tower until it was found:
    /// HuggingFace's "gelu" is the exact erf form, "gelu_new" the tanh
    /// approximation. Collapsing them looks fine and moves every embedding.
    #[test]
    fn gelu_and_gelu_new_are_different_activations() {
        let exact = ClipConfig::from_json(&config_json("gelu", "gelu")).unwrap();
        assert_eq!(exact.activation().unwrap(), Activation::Gelu);

        let approx = ClipConfig::from_json(&config_json("gelu_new", "gelu_new")).unwrap();
        assert_eq!(approx.activation().unwrap(), Activation::GeluNew);
        assert_ne!(
            exact.activation().unwrap(),
            approx.activation().unwrap(),
            "the two gelus must not collapse onto one variant"
        );
    }

    #[test]
    fn openai_checkpoints_ask_for_quick_gelu() {
        let c = ClipConfig::from_json(&config_json("quick_gelu", "quick_gelu")).unwrap();
        assert_eq!(c.activation().unwrap(), Activation::QuickGelu);
        assert_eq!(c.text_activation().unwrap(), Activation::QuickGelu);
    }

    /// The towers are configured independently and can genuinely differ.
    #[test]
    fn each_tower_reads_its_own_activation() {
        let c = ClipConfig::from_json(&config_json("quick_gelu", "gelu")).unwrap();
        assert_eq!(c.activation().unwrap(), Activation::QuickGelu);
        assert_eq!(c.text_activation().unwrap(), Activation::Gelu);
    }

    #[test]
    fn an_unknown_activation_is_an_error_not_a_default() {
        let c = ClipConfig::from_json(&config_json("mish", "gelu")).unwrap();
        let err = c.activation().unwrap_err().to_string();
        assert!(err.contains("mish"), "error should name it: {err}");
    }

    /// A vision-only export has no text tower, and asking for one must say so
    /// rather than panic or invent defaults.
    #[test]
    fn a_vision_only_checkpoint_reports_a_missing_text_tower() {
        let json = r#"{"vision_config": {
            "hidden_size": 768, "intermediate_size": 3072,
            "num_hidden_layers": 12, "num_attention_heads": 12,
            "image_size": 224, "patch_size": 16
        }}"#;
        let c = ClipConfig::from_json(json).unwrap();
        assert!(c.text_config.is_none());
        let err = c.text().unwrap_err().to_string();
        assert!(err.contains("vision-only"), "unhelpful error: {err}");
        // A different patch size changes the whole sequence length.
        assert_eq!(c.patches_per_side(), 14);
        assert_eq!(c.num_positions(), 197);
    }

    #[test]
    fn metadata_describes_the_vision_tower_for_the_shared_encoder() {
        let c = ClipConfig::from_json(&config_json("gelu", "gelu")).unwrap();
        let m = c.metadata();
        assert_eq!(m.hidden_size, 768);
        assert_eq!(m.num_layers, 12);
        assert_eq!(m.head_dim, 64, "768 / 12");
        assert_eq!(m.num_kv_heads, m.num_attention_heads, "CLIP has no GQA");
        assert!(m.is_prenorm, "CLIP norms before each sublayer, unlike BERT");
        assert!(
            m.normalize_embedding,
            "pre_layrnorm runs on the patch sequence"
        );
        assert_eq!(m.max_seq_len, 50, "positions, not tokens");
        assert_eq!(m.vocab_size, 0, "a vision tower has no vocabulary");
        assert!(m.rope_theta.is_none(), "positions are learned, not rotary");
    }

    /// The layout drives every weight lookup, so a typo here fails at load with a
    /// missing-tensor error. Including HuggingFace's own misspelling.
    #[test]
    fn layout_matches_the_checkpoint_tensor_names() {
        let c = ClipConfig::from_json(&config_json("gelu", "gelu")).unwrap();
        let l = c.layout();
        let enc = l.encoder.as_ref().expect("vision layout");

        assert_eq!(
            enc.embedding_norm_weight.as_deref(),
            Some("vision_model.pre_layrnorm.weight"),
            "the missing 'e' is upstream and ships in every CLIP export"
        );
        assert_eq!(
            enc.final_norm_weight.as_deref(),
            Some("vision_model.post_layernorm.weight")
        );
        assert!(
            enc.layer.self_attn.q_weight.contains("{}"),
            "layer index placeholder"
        );
        assert_eq!(
            enc.layer.self_attn.q_weight,
            "vision_model.encoder.layers.{}.self_attn.q_proj.weight"
        );
        assert!(
            enc.layer.ffn.gate_weight.is_none(),
            "CLIP's MLP is not gated"
        );
        assert!(l.decoder.is_none(), "the vision tower is encoder-only");
    }
}
