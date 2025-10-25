//! A generic, backend-agnostic transformer decoder implementation.
//!
//! This module provides `TransformerDecoder`, a reusable component that can represent
//! various decoder-only models like GPT-2, GPT-J, etc. It is designed to be
//! backend-aware, containing either a CPU or GPU implementation.
//!
//! The decoder is constructed generically by relying on the `DecoderArchitecture`
//! trait, which provides the specific weight names and hyperparameters for a
//! given model, allowing for maximum code reuse.

mod cpu;
mod gpu;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::sync::Arc;
pub use crate::{Cache, CpuKVCache, GpuKVCache};
use crate::traits::{
    Device, Decoder, DecoderArchitecture, DecoderOutput, Model, ModelConfig, TransformerConfig,
};
use crate::weights::ModelWeights;
use crate::wgpu_context::WgpuContext;
use cpu::CpuTransformerDecoder;
use gpu::GpuTransformerDecoder;

/// A generic, backend-agnostic transformer decoder stack.
///
/// This enum acts as a container for the backend-specific implementation
/// (e.g., `CpuTransformerDecoder`, `GpuTransformerDecoder`). It dispatches calls to the appropriate
/// backend, providing a single, consistent API for any decoder model.
pub enum TransformerDecoder {
    Cpu(CpuTransformerDecoder),
    Gpu(GpuTransformerDecoder),
}

impl TransformerDecoder {
    /// Creates a new generic `TransformerDecoder` for the specified device.
    ///
    /// This factory function is generic over any configuration `C` that implements
    /// the `DecoderArchitecture` trait. It uses the trait to dynamically load the
    /// correct weights and build the model stack for either the CPU or GPU backend.
    pub fn new<C>(
        weights: &ModelWeights,
        config: Arc<C>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self>
    where
        C: DecoderArchitecture + Send + Sync + 'static,
    {
        match device {
            Device::Cpu => Ok(Self::Cpu(CpuTransformerDecoder::new(weights, config.clone())?)),
            Device::Wgpu => {
                let ctx = context.ok_or_else(|| {
                    anyhow!("A WGPU context is required to create a GPU-based decoder.")
                })?;
                Ok(Self::Gpu(GpuTransformerDecoder::new(weights, config.clone(), ctx)?))
            }
        }
    }
}

/// Implements the base `Model` trait for the generic decoder, delegating to the backend.
impl Model for TransformerDecoder {
    fn device(&self) -> Device {
        match self {
            Self::Cpu(model) => model.device(),
            Self::Gpu(model) => model.device(),
        }
    }
}

/// Implements the `Decoder` trait for the generic decoder, delegating to the backend.
#[async_trait]
impl Decoder for TransformerDecoder {
    type Input = Array2<f32>;
    type Output = DecoderOutput;

    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        match self {
            Self::Cpu(model) => model.forward(input, attention_mask, cache).await,
            Self::Gpu(model) => model.forward(input, attention_mask, cache).await,
        }
    }
}