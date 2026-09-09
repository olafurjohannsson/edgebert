//! Encoder pipeline for embedding and classification models.

use anyhow::{Result, anyhow};
use std::sync::Arc;

use crate::LoadedEmbeddings;
use crate::WgpuContext;
use crate::cpu::encoder::{
    classifier::CpuSequenceClassificationHead,
    config::PoolingStrategy,
    traits::{CpuEncoder, GpuEncoder},
};
use crate::execution::ExecutionPlan;
use crate::traits::Device;

// Pipeline Configuration

/// Configuration for the encoder pipeline.
#[derive(Debug, Clone)]
pub struct EncoderPipelineConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_seq_length: usize,
    pub pooling_strategy: PoolingStrategy,
    /// Whether this pipeline has a classification head
    pub has_head: bool,
    /// Number of output classes (if has_head is true)
    pub num_labels: Option<usize>,
}

/// A container that holds all components needed for encoder inference.
pub struct EncoderPipeline {
    // Core components
    embeddings: LoadedEmbeddings,
    cpu_encoder: Option<Box<dyn CpuEncoder>>,
    gpu_encoder: Option<Box<dyn GpuEncoder>>,

    // Optional classification head (None for SentenceEncoder)
    cpu_head: Option<CpuSequenceClassificationHead>,
    // gpu_head: Option<GpuSequenceClassificationHead>, // TODO:

    // Configuration
    plan: ExecutionPlan,
    config: EncoderPipelineConfig,

    // Runtime context
    context: Option<Arc<WgpuContext>>,
}

impl EncoderPipeline {
    /// Creates a new encoder pipeline.
    pub fn new(
        embeddings: LoadedEmbeddings,
        cpu_encoder: Option<Box<dyn CpuEncoder>>,
        gpu_encoder: Option<Box<dyn GpuEncoder>>,
        cpu_head: Option<CpuSequenceClassificationHead>,
        plan: ExecutionPlan,
        context: Option<Arc<WgpuContext>>,
        config: EncoderPipelineConfig,
    ) -> Result<Self> {
        assert!(cpu_encoder.is_some() || gpu_encoder.is_some());
        let pipeline = Self {
            embeddings,
            cpu_encoder,
            gpu_encoder,
            cpu_head,
            plan,
            context,
            config,
        };

        pipeline.validate_plan(&pipeline.plan)?;
        Ok(pipeline)
    }

    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    pub fn set_plan(&mut self, plan: ExecutionPlan) -> Result<()> {
        self.validate_plan(&plan)?;
        self.plan = plan;
        Ok(())
    }

    fn validate_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        match plan.embeddings {
            Device::Cpu if !self.embeddings.is_cpu() => {
                return Err(anyhow!("Plan requires CPU embeddings but not loaded"));
            }
            // `is_gpu` is native-only; on wasm the plan can never select Wgpu because
            // no GPU context can be constructed.
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
            Device::Wgpu if !self.embeddings.is_gpu() => {
                return Err(anyhow!("Plan requires GPU embeddings but not loaded"));
            }
            _ => {}
        }

        match plan.layers {
            Device::Cpu if self.cpu_encoder.is_none() => {
                return Err(anyhow!("Plan requires CPU encoder but not loaded"));
            }
            Device::Wgpu if self.gpu_encoder.is_none() => {
                return Err(anyhow!("Plan requires GPU encoder but not loaded"));
            }
            _ => {}
        }

        if plan.needs_gpu() && self.context.is_none() {
            return Err(anyhow!("Plan requires GPU but no WgpuContext available"));
        }

        if plan.needs_cpu() && self.cpu_encoder.is_none() {
            return Err(anyhow!("Plan requires CPU encoder but not loaded"));
        }

        Ok(())
    }

    pub fn embeddings(&self) -> &LoadedEmbeddings {
        &self.embeddings
    }

    pub fn cpu_encoder(&self) -> Option<&dyn CpuEncoder> {
        self.cpu_encoder.as_ref().map(|e| e.as_ref())
    }

    pub fn gpu_encoder(&self) -> Option<&dyn GpuEncoder> {
        self.gpu_encoder.as_ref().map(|e| e.as_ref())
    }

    pub fn cpu_head(&self) -> Option<&CpuSequenceClassificationHead> {
        self.cpu_head.as_ref()
    }

    pub fn context(&self) -> Option<&Arc<WgpuContext>> {
        self.context.as_ref()
    }

    pub fn has_head(&self) -> bool {
        self.cpu_head.is_some()
    }

    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn max_seq_length(&self) -> usize {
        self.config.max_seq_length
    }

    pub fn pooling_strategy(&self) -> PoolingStrategy {
        self.config.pooling_strategy.clone()
    }

    pub fn num_labels(&self) -> Option<usize> {
        self.cpu_head.as_ref().map(|h| h.num_classes())
    }

    pub fn labels(&self) -> Option<&[String]> {
        self.cpu_head.as_ref().and_then(|h| h.labels())
    }
}
