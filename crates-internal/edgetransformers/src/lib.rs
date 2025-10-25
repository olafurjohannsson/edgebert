//! Core transformer components for building transformer-based models
//!
//! This crate provides the fundamental building blocks for transformer architectures
//! without model-specific implementations.

pub mod activations;
pub mod attention;
pub mod bind_group;
pub mod cache;
// pub mod config;
pub mod decoder;
pub mod embeddings;
pub mod encoder;
pub mod feedforward;
pub mod gpu_ops;
pub mod gpu_pipeline;
pub mod layer_norm;
pub mod model_type;
pub mod pooling;
pub mod prelude;
pub mod traits;
pub mod utils;
pub mod weights;
pub mod wgpu_context;
pub mod wgpu_ops;

// Re-export
pub use activations::{Activation, gelu, softmax};
pub use attention::MultiHeadAttention;
pub use cache::{CpuKVCache, GpuKVCache};
// pub use config::TransformerConfig;
pub use traits::TransformerConfig;
pub use embeddings::{EmbeddingConfig, Embeddings};
pub use feedforward::FeedForward;
pub use layer_norm::LayerNorm;
pub use pooling::{PoolingStrategy, cls_pool, mean_pool};
pub use traits::Cache;
pub use utils::linear_algebra::{apply_attention_mask, matmul_3d_2d, matmul_4d};

use anyhow::Result;
use ndarray::{Array2, Array3};

/// Base trait for transformer models
pub trait TransformerModel {
    /// Forward pass through the transformer
    fn forward(&self, input_ids: &Array2<f32>, attention_mask: &Array2<f32>)
    -> Result<Array3<f32>>;

    /// Get the model configuration
    fn config(&self) -> &dyn TransformerConfig;
}

/// Trait for encoder models that produce embeddings
pub trait Encoder {
    /// Encode text into embeddings
    fn encode(&self, texts: Vec<&str>, normalize: bool) -> Result<Vec<Vec<f32>>>;
}

/// A generic transformer layer combining attention and feedforward
pub struct TransformerLayer {
    pub attention: MultiHeadAttention,
    pub feedforward: FeedForward,
    pub layer_norm1: LayerNorm,
    pub layer_norm2: LayerNorm,
}

impl TransformerLayer {
    pub fn forward(
        &self,
        hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
    ) -> Result<Array3<f32>> {
        // // Self attention with residual
        // let mut attention_out = self.attention.forward(&input, None, Some(attention_mask))?;
        // attention_out += &hidden;
        // let attention_out = self.layer_norm1.forward_3d(&attention_out);

        // // Feed forward with residual
        // let mut ff_out = self.feedforward.forward(&attention_out)?;
        // ff_out += &attention_out;
        // let output = self.layer_norm2.forward_3d(&ff_out);

        // Ok(output)
        
        let is_prenorm = config.is_prenorm();
        let is_causal = config.is_causal();

        // Choose architecture: pre-norm (GPT) vs post-norm (BERT)
        if is_prenorm {
            self.forward_prenorm(hidden, attention_mask, is_causal)
        } else {
            self.forward_postnorm(hidden, attention_mask, is_causal)
        }
    }
    fn forward_prenorm(
        &self,
        mut hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        is_causal: bool,
    ) -> Result<Array3<f32>> {
        // Pre-norm: LayerNorm → Attention → Residual
        let ln1_out = self.layer_norm1.forward_3d(&hidden);
        let (attn_out, _) =
            self.attention
                .forward_bart(&ln1_out, None, Some(attention_mask), is_causal, None)?;
        hidden = hidden + attn_out;

        // Pre-norm: LayerNorm → FFN → Residual
        let ln2_out = self.layer_norm2.forward_3d(&hidden);
        let ffn_out = self.feedforward.forward(&ln2_out)?;
        hidden = hidden + ffn_out;

        Ok(hidden)
    }

    fn forward_postnorm(
        &self,
        mut hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        is_causal: bool,
    ) -> Result<Array3<f32>> {
        // Post-norm: Attention → Residual → LayerNorm
        let (attn_out, _) =
            self.attention
                .forward_bart(&hidden, None, Some(attention_mask), is_causal, None)?;
        hidden = hidden + attn_out;
        hidden = self.layer_norm1.forward_3d(&hidden);

        // Post-norm: FFN → Residual → LayerNorm
        let ffn_out = self.feedforward.forward(&hidden)?;
        hidden = hidden + ffn_out;
        hidden = self.layer_norm2.forward_3d(&hidden);

        Ok(hidden)
    }
}
