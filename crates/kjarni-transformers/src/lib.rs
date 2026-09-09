//! # Kjarni Transformers
//!
//! Transformer inference for Rust.
//!
//! **Author:** Ólafur Aron Jóhannsson <olafurjohannss@gmail.com>  
//! **License:** MIT OR Apache-2.0

pub mod activations;
pub mod audio;
pub mod cache;
pub mod chat;

pub mod cpu;

pub mod execution;

pub mod common;
pub mod decoder;
#[cfg(not(target_arch = "wasm32"))]
pub mod encoder_decoder;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
pub mod gpu;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
pub mod gpu_ops;

pub mod linear_layer;
pub mod loaders;
pub mod models;
pub mod pipeline;
pub mod pooling;
pub mod stats;
pub mod tensor;
pub mod tokenizer;
pub mod traits;
pub mod utils;
pub mod weights;

pub mod rope {
    pub use crate::cpu::rope::RoPE;
}

pub mod normalization {
    pub use crate::cpu::normalization::{LayerNorm, Normalization, RMSNorm};
}

pub mod feedforward {
    pub use crate::cpu::feedforward::{
        FeedForward, LegacyFeedForward, StdFeedForward, StdFeedForwardNew, SwiGluFeedForward,
    };
}

pub mod attention {
    pub use crate::cpu::attention::multi_head_attention::MultiHeadAttention;
}

pub use audio::{
    AudioConvFrontend, AudioData, AudioLoaderConfig, AudioPipeline, MelConfig, create_sine_wave,
    load_audio, load_audio_bytes, load_audio_for_whisper,
};

pub use crate::{
    chat::templates::{ChatTemplate, Conversation, Message, Role},
    cpu::{
        attention::multi_head_attention::MultiHeadAttention,
        embeddings::{EmbeddingData, Embeddings},
        normalization::Normalization,
    },
    execution::ExecutionPlan,
    feedforward::FeedForward,
    pooling::{PoolingStrategy, cls_pool, last_token_pool, max_pool, mean_pool},
};

// `loaders` itself builds on wasm; only the decoder pipeline does not. Re-exporting
// them separately is what lets the encoder pipeline compile for the browser.
pub use crate::loaders::{
    EmbeddingConfig, EmbeddingConfigBuilder, EmbeddingInput, LMHeadConfig, LoadedEmbeddings,
    LoadedLMHead,
};

#[cfg(not(target_arch = "wasm32"))]
pub use crate::pipeline::{DecoderPipeline, DecoderPipelineConfig};

#[cfg(target_arch = "wasm32")]
pub mod prelude {
    pub use crate::cache::{Cache, CpuKVCache};
    pub use crate::models::LanguageModel;
    pub use crate::traits::Device;
}

pub use cache::{Cache, CpuKVCache};

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
pub use gpu_ops::context::WgpuContext;

/// Placeholder GPU context for wasm builds.
///
/// Uninhabited on purpose: the encoder pipeline stores `Option<Arc<WgpuContext>>`,
/// and an empty enum lets that field exist while making a `Some(..)` value
/// impossible to construct. See `cpu::encoder::traits::GpuEncoder` for the same
/// reasoning applied to the encoder traits.
#[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
pub enum WgpuContext {}

pub use models::{LanguageModel, ModelArchitecture, ModelType};
pub use traits::Device;

#[cfg(not(target_arch = "wasm32"))]
pub mod prelude {
    pub use crate::cache::{Cache, CpuKVCache};
    pub use crate::gpu::cache::{GpuBeamKVCache, GpuKVCache};
    pub use crate::gpu_ops::context::WgpuContext;
    pub use crate::models::LanguageModel;
    pub use crate::traits::Device;
}

#[cfg(test)]
pub mod tests;

#[global_allocator]
static GLOBAL: crate::utils::alloc_stats::TracingAllocator =
    crate::utils::alloc_stats::TracingAllocator;
