pub mod backend;
pub mod generator;
pub mod traits;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
mod gpu;

pub mod prelude {
    pub use crate::cpu::decoder::{
        CpuDecoderBackend, CpuRoPEDecoderLayer, DecoderAttention, DecoderLayer,
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub use crate::decoder::gpu::{GpuPreNormDecoderLayer, GpuRoPEDecoderLayer};
    pub use crate::decoder::{
        backend::AnyDecoderBackend,
        generator::DecoderGenerator,
        traits::{
            CpuDecoder, CpuDecoderOps, DecoderGenerationBackend, DecoderLanguageModel, GpuDecoder,
            GpuDecoderOps,
        },
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub use crate::gpu::decoder::backend::GpuDecoderBackend;
}

#[cfg(test)]
mod test_generator;
