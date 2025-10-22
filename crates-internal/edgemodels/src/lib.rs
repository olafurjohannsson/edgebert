//! GPT implementation using edgeTransformers
//! 
//! Provides autoregressive language models for text generation.

pub mod config;
pub mod model;
pub mod weights;
pub mod tokenizer;
pub mod generation;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-exports
pub use config::GPTConfig;
pub use model::{GenerativeModel, ModelType};
pub use weights::ModelWeights;
pub use generation::{GenerationConfig, SamplingStrategy};

#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::wasm::BPETokenizer;