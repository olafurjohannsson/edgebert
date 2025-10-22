//! BERT implementation using edgeTransformers
//! 
//! Provides bi-encoder and cross-encoder models for text embeddings and similarity scoring.

pub mod config;
pub mod model;
pub mod weights;
pub mod tokenizer;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-exports
pub use config::BertConfig;
pub use model::{bert::BertModel, bert::BertModelType};
pub use weights::ModelWeights;

#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::wasm::WordPieceTokenizer;

// Re-export commonly used edgeTransformers utilities
pub use edgetransformers::utils::linear_algebra::cosine_similarity;