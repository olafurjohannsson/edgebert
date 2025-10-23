//! Core transformer components for building transformer-based models
//! 
//! This crate provides the fundamental building blocks for transformer architectures
//! without model-specific implementations.

pub mod attention;
pub mod feedforward;
pub mod layer_norm;
pub mod embeddings;
pub mod activations;
pub mod pooling;
pub mod utils;
pub mod config;
pub mod traits;
pub mod weights;
pub mod encoder;
pub mod prelude;

pub mod wgpu_context;
pub mod wgpu_ops;
pub mod gpu_pipeline;

// Re-export
pub use attention::MultiHeadAttention;
pub use feedforward::FeedForward;
pub use layer_norm::LayerNorm;
pub use embeddings::{Embeddings, EmbeddingConfig};
pub use activations::{gelu, softmax, Activation};
pub use pooling::{mean_pool, cls_pool, PoolingStrategy};
pub use config::TransformerConfig;

pub use utils::linear_algebra::{
    matmul_3d_2d, 
    matmul_4d, 
    apply_attention_mask
};

use anyhow::Result;
use ndarray::{Array2, Array3};

/// Base trait for transformer models
pub trait TransformerModel {
    /// Forward pass through the transformer
    fn forward(
        &self, 
        input_ids: &Array2<f32>, 
        attention_mask: &Array2<f32>
    ) -> Result<Array3<f32>>;
    
    /// Get the model configuration
    fn config(&self) -> &dyn TransformerConfig;
}

/// Trait for encoder models that produce embeddings
pub trait Encoder {
    /// Encode text into embeddings
    fn encode(
        &self, 
        texts: Vec<&str>, 
        normalize: bool
    ) -> Result<Vec<Vec<f32>>>;
}

/// A generic transformer layer combining attention and feedforward
pub struct TransformerLayer {
    pub attention: MultiHeadAttention,
    pub feedforward: FeedForward,
    pub layer_norm1: LayerNorm,
    pub layer_norm2: LayerNorm,
}

impl TransformerLayer {
    pub fn forward(&self, input: Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array3<f32>> {
        // Self attention with residual
        let mut attention_out = self.attention.forward(&input, None, Some(attention_mask))?;
        attention_out += &input;
        let attention_out = self.layer_norm1.forward_3d(&attention_out);
        
        // Feed forward with residual
        let mut ff_out = self.feedforward.forward(&attention_out)?;
        ff_out += &attention_out;
        let output = self.layer_norm2.forward_3d(&ff_out);
        
        Ok(output)
    }
}