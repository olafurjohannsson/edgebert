pub mod attention;
pub mod cache;
pub mod encoder;
pub mod ffn;
pub mod ffn_swiglu;
pub mod layers;
pub mod rope;
use crate::gpu::{GpuTensor, GpuTensorPool};

pub use ffn::{GpuFeedForwardStd, GpuFeedForwardWeights as GpuFeedForwardWeightsStd};
pub use ffn_swiglu::{GpuSwiGLUFFN, GpuSwiGLUFFNWeights};

pub enum GpuFeedForwardWeights {
    Standard(GpuFeedForwardWeightsStd),
    SwiGLU(GpuSwiGLUFFNWeights),
}

impl GpuFeedForwardWeights {
    pub fn up_proj_shape(&self) -> &[usize] {
        match self {
            GpuFeedForwardWeights::SwiGLU(s) => s.up_proj.shape(),
            _ => unimplemented!(),
        }
    }
    pub fn down_proj_shape(&self) -> &[usize] {
        match self {
            GpuFeedForwardWeights::SwiGLU(s) => s.down_proj.shape(),
            _ => unimplemented!(),
        }
    }
    pub fn gate_proj_shape(&self) -> &[usize] {
        match self {
            GpuFeedForwardWeights::SwiGLU(s) => s.gate_proj.shape(),
            _ => unimplemented!(),
        }
    }
}
#[allow(clippy::large_enum_variant)] // built once per model; boxing would cost a
// dereference inside the per-layer encode path to save stack that is never hot.
pub enum GpuFeedForward {
    Standard(GpuFeedForwardStd),
    SwiGLU(GpuSwiGLUFFN),
}

impl GpuFeedForward {
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &GpuFeedForwardWeights,
        input: &GpuTensor,
        output: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) {
        match (self, weights) {
            (Self::Standard(ffn), GpuFeedForwardWeights::Standard(w)) => {
                ffn.encode_2(encoder, input, w, pool, output)
            }
            (Self::SwiGLU(swi), GpuFeedForwardWeights::SwiGLU(w)) => {
                swi.encode(encoder, w, input, output, pool)
            }
            _ => panic!("FFN type mismatch"),
        }
    }
}
