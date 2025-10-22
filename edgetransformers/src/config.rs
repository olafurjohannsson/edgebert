//! Configuration traits and structs for transformer models

use serde::{Deserialize, Serialize};

/// Base trait for transformer configurations
pub trait TransformerConfig {
    fn hidden_size(&self) -> usize;
    fn num_attention_heads(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn max_position_embeddings(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn intermediate_size(&self) -> usize;
    fn layer_norm_eps(&self) -> f32;
    fn hidden_dropout_prob(&self) -> f32;
    fn attention_dropout_prob(&self) -> f32;
}

/// Common configuration structure that can be shared across models
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BaseConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    #[serde(default = "default_dropout")]
    pub hidden_dropout_prob: f32,
    #[serde(default = "default_dropout")]
    pub attention_dropout_prob: f32,
    pub hidden_act: String,
    pub model_type: String,
}

fn default_dropout() -> f32 {
    0.1
}

impl TransformerConfig for BaseConfig {
    fn hidden_size(&self) -> usize { self.hidden_size }
    fn num_attention_heads(&self) -> usize { self.num_attention_heads }
    fn num_hidden_layers(&self) -> usize { self.num_hidden_layers }
    fn max_position_embeddings(&self) -> usize { self.max_position_embeddings }
    fn vocab_size(&self) -> usize { self.vocab_size }
    fn intermediate_size(&self) -> usize { self.intermediate_size }
    fn layer_norm_eps(&self) -> f32 { self.layer_norm_eps }
    fn hidden_dropout_prob(&self) -> f32 { self.hidden_dropout_prob }
    fn attention_dropout_prob(&self) -> f32 { self.attention_dropout_prob }
}