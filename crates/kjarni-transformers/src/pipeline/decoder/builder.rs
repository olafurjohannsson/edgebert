use crate::WgpuContext;
use crate::decoder::prelude::{CpuDecoder, GpuDecoder};
use crate::execution::ExecutionPlan;
use crate::loaders::{LMHeadConfig, LoadedLMHead};
use crate::models::base::ModelLoadConfig;
use crate::pipeline::{DecoderPipeline, DecoderPipelineConfig};
use crate::traits::{Device, ModelConfig};
use crate::weights::ModelWeights;
use crate::{EmbeddingConfig, LoadedEmbeddings};
use anyhow::{Result, anyhow};
use std::sync::Arc;
pub struct DecoderPipelineBuilder<'a> {
    weights: &'a ModelWeights,
    config: Arc<dyn ModelConfig>,
    load_config: ModelLoadConfig,
    context: Option<Arc<WgpuContext>>,
    cpu_backend: Option<Box<dyn CpuDecoder>>,
    gpu_backend: Option<Box<dyn GpuDecoder>>,
}

impl<'a> DecoderPipelineBuilder<'a> {
    pub fn new(weights: &'a ModelWeights, config: Arc<dyn ModelConfig>) -> Self {
        Self {
            weights,
            config,
            load_config: ModelLoadConfig::default(),
            context: None,
            cpu_backend: None,
            gpu_backend: None,
        }
    }

    pub fn with_load_config(mut self, cfg: ModelLoadConfig) -> Self {
        self.load_config = cfg;
        self
    }

    pub fn with_context_opt(mut self, ctx: Option<Arc<WgpuContext>>) -> Self {
        self.context = ctx;
        self
    }

    pub fn with_backends(
        mut self,
        cpu: Option<Box<dyn CpuDecoder>>,
        gpu: Option<Box<dyn GpuDecoder>>,
    ) -> Self {
        self.cpu_backend = cpu;
        self.gpu_backend = gpu;
        self
    }

    pub fn build(self) -> Result<DecoderPipeline> {
        let meta = self.config.metadata();
        let layout = self.config.layout();
        let ctx = self.context.as_ref();
        let target_dt = self.load_config.target_dtype;
        // Determine Device Strategy
        let primary_device = if ctx.is_some() {
            Device::Wgpu
        } else {
            Device::Cpu
        };
        let plan = ExecutionPlan::from_load_config(primary_device, &self.load_config);

        // Extract Decoder Layout
        let dec_layout = layout
            .decoder
            .as_ref()
            .ok_or_else(|| anyhow!("Pipeline requires a DecoderLayout in ModelLayout"))?;

        let start_ram = crate::utils::alloc_stats::get_current_ram_usage_mb();
        log::debug!("[Builder] Pre-Embeddings RAM: {:.2} MB", start_ram);

        // Load Embeddings
        let mut emb_builder = EmbeddingConfig::builder(&layout.token_embedding, meta.hidden_size);
        if let Some(pos) = &dec_layout.position_embedding {
            emb_builder = emb_builder.position_embedding(pos);
        }
        if let Some(tok) = &dec_layout.token_type_embedding {
            emb_builder = emb_builder.type_embedding(tok);
        }
        let tied_weights = layout.lm_head == layout.token_embedding;
        let emb_load_cpu =
            plan.embeddings == Device::Cpu || (tied_weights && plan.lm_head == Device::Cpu);
        let emb_load_gpu =
            plan.embeddings == Device::Wgpu || (tied_weights && plan.lm_head == Device::Wgpu);
        // The wasm constructor takes no context and no device flags: CPU is the only
        // path there.
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
        let embeddings = LoadedEmbeddings::new(
            ctx,
            self.weights,
            emb_builder
                .position_offset(meta.extra_pos_embeddings)
                .scale_embeddings(meta.scale_embeddings)
                .build(),
            emb_load_cpu,
            emb_load_gpu,
            target_dt,
        )?;
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
        let embeddings = {
            let _ = (emb_load_cpu, emb_load_gpu);
            LoadedEmbeddings::new(
                self.weights,
                emb_builder
                    .position_offset(meta.extra_pos_embeddings)
                    .scale_embeddings(meta.scale_embeddings)
                    .build(),
                target_dt,
            )?
        };

        let mid_ram = crate::utils::alloc_stats::get_current_ram_usage_mb();
        log::debug!(
            "[Builder] Post-Embeddings RAM: {:.2} MB (Delta: {:.2} MB)",
            mid_ram,
            mid_ram - start_ram
        );

        log::debug!(
            "LM head quantization: {:?}",
            self.load_config.quantize_lm_head
        );

        // Tied weights alias the embedding tensors, which needs the GPU-aware
        // constructor; on wasm the head is simply loaded from the weights.
        // Tying normally aliases the embedding tensors, which costs nothing. But a
        // quantised checkpoint dequantises its embedding table for the lookup, and
        // aliasing that hands the LM head a vocab-sized F32 matmul: on a 3B Q4_K_M
        // model that is 1.57GB streamed per token, measured at 77ms, 37% of CPU
        // decode. When the tensor is still quantised on disk the head is loaded from
        // it directly instead. That is one extra copy of the quantised table, which
        // is small precisely because it is quantised.
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
        let tied_head_is_quantised = tied_weights
            && self
                .weights
                .tensor_dtype(&layout.lm_head)
                .is_ok_and(|dt| dt.is_quantized());

        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
        let lm_head = if tied_weights && !tied_head_is_quantised {
            log::info!("Using tied weights between embeddings and LM head");
            LoadedLMHead::from_shared_weights(
                ctx,
                embeddings.word_embeddings_cpu(),
                None,
                embeddings.word_embeddings_gpu(),
                LMHeadConfig::new(&layout.lm_head, meta.vocab_size, meta.hidden_size),
                self.load_config.quantize_lm_head,
            )?
        } else {
            if tied_head_is_quantised {
                log::info!(
                    "LM head shares a tensor with the embeddings but loads it quantised, \
                     to avoid a vocab-sized F32 matmul every token"
                );
            }
            LoadedLMHead::new(
                ctx,
                self.weights,
                None,
                LMHeadConfig::new(&layout.lm_head, meta.vocab_size, meta.hidden_size),
                plan.lm_head == Device::Cpu,
                plan.lm_head == Device::Wgpu,
                self.load_config.quantize_lm_head,
            )?
        };
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
        let lm_head = LoadedLMHead::new(
            self.weights,
            None,
            LMHeadConfig::new(&layout.lm_head, meta.vocab_size, meta.hidden_size),
            target_dt,
        )?;
        let end_ram = crate::utils::alloc_stats::get_current_ram_usage_mb();
        log::debug!(
            "[Builder] Post-LMHead RAM: {:.2} MB (Delta: {:.2} MB)",
            end_ram,
            end_ram - mid_ram
        );

        DecoderPipeline::new(
            embeddings,
            self.cpu_backend,
            self.gpu_backend,
            lm_head,
            plan,
            self.context.clone(),
            DecoderPipelineConfig {
                num_layers: meta.num_layers,
                hidden_size: meta.hidden_size,
                vocab_size: meta.vocab_size,
                max_sequence_length: self.load_config.max_sequence_length,
                max_batch_size: self.load_config.max_batch_size,
            },
        )
    }
}
