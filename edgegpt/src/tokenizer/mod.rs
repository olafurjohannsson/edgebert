//! Tokenizer implementations for GPT

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers::Tokenizer;

