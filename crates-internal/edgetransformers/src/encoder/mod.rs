//! A generic, backend-agnostic transformer encoder implementation.
//!
//! This module provides `TransformerEncoder`, a reusable component that can represent
//! various encoder-only models like BERT, RoBERTa, etc. It is designed to be
//! backend-aware, containing either a CPU or a (future) GPU implementation.
//!
//! The encoder is constructed generically by relying on the `EncoderArchitecture`
//! trait, which provides the specific weight names and hyperparameters for a

//! given model, allowing for maximum code reuse.

mod cpu;
mod gpu;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::sync::Arc;

use crate::traits::{
    Device, Encoder, EncoderArchitecture, EncoderOutput, Model, ModelConfig, TransformerConfig,
};
use crate::weights::ModelWeights;
use crate::wgpu_context::WgpuContext;
use crate::{Embeddings, FeedForward, LayerNorm, MultiHeadAttention, TransformerLayer};
use cpu::CpuTransformerEncoder;
use gpu::GpuTransformerEncoder;

/// A generic, backend-agnostic transformer encoder stack.
///
/// This enum acts as a container for the backend-specific implementation
/// (e.g., `CpuTransformerEncoder`). It dispatches calls to the appropriate
/// backend, providing a single, consistent API for any encoder model.
pub enum TransformerEncoder {
    Cpu(CpuTransformerEncoder),
    Gpu(GpuTransformerEncoder),
}

impl TransformerEncoder {
    /// Creates a new generic `TransformerEncoder` for the specified device.
    ///
    /// This factory function is generic over any configuration `C` that implements
    /// the `EncoderArchitecture` trait. It uses the trait to dynamically load the
    /// correct weights and build the model stack for either the CPU or GPU backend.
    pub fn new<C>(
        weights: &ModelWeights,
        config: Arc<C>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self>
    where
        C: EncoderArchitecture + Send + Sync + 'static,
    {
        match device {
            Device::Cpu => Ok(Self::Cpu(CpuTransformerEncoder::new(weights, config.clone())?)),
            Device::Wgpu => {
                let ctx = context.ok_or_else(|| {
                    anyhow!("A WGPU context is required to create a GPU-based encoder.")
                })?;
                Ok(Self::Gpu(GpuTransformerEncoder::new(weights, config.clone(), ctx)?))
            }
        }
    }
}

/// Implements the base `Model` trait for the generic encoder, delegating to the backend.
impl Model for TransformerEncoder {
    fn device(&self) -> Device {
        match self {
            Self::Cpu(model) => model.device(),
            Self::Gpu(model) => model.device(),
        }
    }
}

/// Implements the `Encoder` trait for the generic encoder, delegating to the backend.
#[async_trait]
impl Encoder for TransformerEncoder {
    type Input = Array2<f32>;
    type Output = EncoderOutput;

    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
    ) -> Result<Self::Output> {
        match self {
            Self::Cpu(model) => model.forward(input, attention_mask).await,
            Self::Gpu(model) => model.forward(input, attention_mask).await,
        }
    }
}
