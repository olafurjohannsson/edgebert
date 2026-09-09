//! LLaMA-style decoder-only language model.
//!
//! This module provides the `LlamaModel`, a model container responsible for loading
//! weights and configuration for Llama and its variants.
//!
//! The actual text generation is handled by the generic `Generator` struct.

pub mod config;
// No GPU backend on wasm.
pub mod cpu_decoder;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
pub mod gpu_decoder;
pub mod model;

pub use config::LlamaConfig;
pub use model::LlamaModel;

#[cfg(test)]
mod tests;
