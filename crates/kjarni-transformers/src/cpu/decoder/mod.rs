mod decoder_attention;
mod decoder_backend;
mod decoder_layer;
mod gqa_projection;
mod rope_decoder_layer;
mod speculation;
pub mod streaming_attention;
pub use crate::cpu::decoder::{
    decoder_attention::DecoderAttention,
    decoder_backend::CpuDecoderBackend,
    decoder_layer::DecoderLayer,
    gqa_projection::GQAProjection,
    rope_decoder_layer::CpuRoPEDecoderLayer,
    speculation::{DraftModelContext, run_speculative_generation_loop},
};

#[cfg(test)]
mod tests;
