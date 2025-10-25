use edgetransformers::traits::{
    EncoderArchitecture, LayerAttentionNames, LayerFeedForwardNames, ModelConfig, TransformerConfig,
};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct RobertaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub layer_norm_eps: f32,
}

impl ModelConfig for RobertaConfig {}

impl TransformerConfig for RobertaConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn layer_norm_eps(&self) -> f32 {
        self.layer_norm_eps
    }
}

impl EncoderArchitecture for RobertaConfig {
    fn transpose_ffn_weights(&self) -> bool {
        false  // RoBERTa doesn't transpose (same as GPT-2 style)
    }

    fn get_embedding_weight_names(&self) -> (&str, &str, &str) {
        (
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "",  // No token_type_embeddings - return empty string
        )
    }

    fn get_embedding_layer_norm_names(&self) -> (&str, &str) {
        ("embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias")
    }

    fn get_attention_names(&self, i: usize) -> LayerAttentionNames {
        let prefix = format!("encoder.layer.{}.attention", i);
        LayerAttentionNames {
            q_weight: format!("{}.self.query.weight", prefix),
            q_bias: format!("{}.self.query.bias", prefix),
            k_weight: format!("{}.self.key.weight", prefix),
            k_bias: format!("{}.self.key.bias", prefix),
            v_weight: format!("{}.self.value.weight", prefix),
            v_bias: format!("{}.self.value.bias", prefix),
            output_weight: format!("{}.output.dense.weight", prefix),
            output_bias: format!("{}.output.dense.bias", prefix),
            norm_weight: format!("{}.output.LayerNorm.weight", prefix),
            norm_bias: format!("{}.output.LayerNorm.bias", prefix),
        }
    }

    fn get_feed_forward_names(&self, i: usize) -> LayerFeedForwardNames {
        let prefix = format!("encoder.layer.{}", i);
        LayerFeedForwardNames {
            intermediate_weight: format!("{}.intermediate.dense.weight", prefix),
            intermediate_bias: format!("{}.intermediate.dense.bias", prefix),
            output_weight: format!("{}.output.dense.weight", prefix),
            output_bias: format!("{}.output.dense.bias", prefix),
            norm_weight: format!("{}.output.LayerNorm.weight", prefix),
            norm_bias: format!("{}.output.LayerNorm.bias", prefix),
        }
    }
}