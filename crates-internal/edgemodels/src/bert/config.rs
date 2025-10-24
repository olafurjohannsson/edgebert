//! Configuration and architectural details for BERT models.
use edgetransformers::traits::{
    EncoderArchitecture, LayerAttentionNames, LayerFeedForwardNames, ModelConfig, TransformerConfig,
};
use serde::Deserialize;

/// Represents the configuration of a BERT model, deserialized from a `config.json` file.
#[derive(Clone, Debug, Deserialize)]
pub struct BertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub layer_norm_eps: f32,
    // Add other fields from your config.json as needed
}

/// Mark `BertConfig` as a valid model configuration.
impl ModelConfig for BertConfig {}

/// Implement the high-level `TransformerConfig` trait.
impl TransformerConfig for BertConfig {
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

/// Implement the `EncoderArchitecture` trait to provide the specific tensor names for BERT.
///
/// This is the "blueprint" that allows the generic `TransformerEncoder` to load
/// a BERT model's weights correctly.
impl EncoderArchitecture for BertConfig {

    fn transpose_ffn_weights(&self) -> bool {
        true
    }

    /// Provides the names for the three embedding tables.
    fn get_embedding_weight_names(&self) -> (&str, &str, &str) {
        (
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
        )
    }

    /// Provides the names for the first LayerNorm after embeddings.
    fn get_embedding_layer_norm_names(&self) -> (&str, &str) {
        ("embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias")
    }

    /// Generates the names for all tensors in a layer's attention block.
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

    /// Generates the names for all tensors in a layer's feed-forward block.
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