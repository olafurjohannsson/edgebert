//! GPT implementation using edgeTransformers
//! 
//! Provides autoregressive language models for text generation.

pub mod bert;
pub mod roberta;
pub mod gptconfig;
pub mod bertconfig;
pub mod model;
pub mod gptweights;
pub mod bertweights;
pub mod tokenizer;
pub mod generation;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-exports
pub use gptconfig::GPTConfig;
pub use model::{GenerativeModel, GenerativeModelType};
pub use gptweights::GPTModelWeights;
pub use generation::{GenerationConfig, SamplingStrategy};

#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::wasm::BPETokenizer;