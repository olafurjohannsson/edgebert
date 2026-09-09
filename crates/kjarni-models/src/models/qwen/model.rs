// Filesystem paths are a native-only concern: the wasm builds load from bytes.
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use kjarni_transformers::Device;
use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;

use crate::models::llama::cpu_decoder::LlamaCpuDecoder;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::models::llama::gpu_decoder::LlamaGpuDecoder;
use crate::models::qwen::config::QwenConfig;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::gpu::{GpuFrameContext, GpuTensor, cache::GpuKVCache};

use kjarni_transformers::{
    ChatTemplate, WgpuContext,
    cache::{Cache, CpuKVCache},
    common::{DecodingStrategy, GenerationConfig, HFGenerationDefaults, SamplingParams},
    decoder::prelude::*,
    loaders::LoadedRoPE,
    models::base::{AutoregressiveLoop, ModelLoadConfig},
    models::{LanguageModel, ModelType},
    pipeline::{DecoderModelFactory, DecoderPipeline},
    traits::{InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

pub struct QwenModel {
    pipeline: DecoderPipeline,
    tokenizer: Tokenizer,
    config: Arc<QwenConfig>,
    chat_template: Option<Box<dyn ChatTemplate>>,
    generation_defaults: Option<HFGenerationDefaults>,
}

impl DecoderModelFactory for QwenModel {
    type Config = QwenConfig;

    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>> {
        QwenConfig::from_loader(weights.loader(), Some(weights.config_json()))
    }

    // On wasm there is no GPU backend, so `context` goes unread and the
    // locals are never reassigned. The signature is shared with native.
    #[cfg_attr(
        target_arch = "wasm32",
        allow(unused_variables, unused_mut, unused_assignments)
    )]
    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        rope: &LoadedRoPE,
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(Option<Box<dyn CpuDecoder>>, Option<Box<dyn GpuDecoder>>)> {
        let mut cpu = None;
        let mut gpu = None;

        if device.is_cpu() || load_config.offload_embeddings {
            cpu = Some(Box::new(LlamaCpuDecoder::new(
                weights,
                meta.clone(),
                layout.clone(),
                rope.cpu.clone(),
                load_config.target_dtype,
            )?) as Box<dyn CpuDecoder>);
        } else if device.is_gpu() {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
            if let Some(ctx) = context {
                gpu = Some(Box::new(LlamaGpuDecoder::new(
                    ctx,
                    weights,
                    meta.clone(),
                    layout.clone(),
                    rope.gpu.clone(),
                    load_config,
                )?) as Box<dyn GpuDecoder>);
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
            return Err(anyhow!("GPU decoding is not available in WebAssembly"));
        } else {
            log::error!("Invalid device in QwenModel");
        }

        Ok((cpu, gpu))
    }

    fn new_from_pipeline(
        pipeline: DecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<QwenConfig>,
        _model_type: Option<ModelType>,
        generation_defaults: Option<HFGenerationDefaults>,
        chat_template: Option<Box<dyn ChatTemplate>>,
    ) -> Self {
        Self {
            pipeline,
            tokenizer,
            config,
            chat_template,
            generation_defaults,
        }
    }
}

impl QwenModel {
    /// Native only: loading from disk or the registry needs a filesystem.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        kjarni_transformers::pipeline::DecoderLoader::load_from_registry::<Self>(
            model_type,
            cache_dir,
            device,
            context,
            load_config,
        )
        .await
    }
    /// Native only: loading from disk or the registry needs a filesystem.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained(
        model_path: &Path,
        device: kjarni_transformers::prelude::Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<ModelLoadConfig>,
        model_tyoe: Option<ModelType>,
    ) -> Result<Self> {
        kjarni_transformers::pipeline::DecoderLoader::load_from_pretrained::<Self>(
            model_path,
            device,
            context,
            decoder_config,
            model_tyoe,
        )
    }

    pub fn config(&self) -> &Arc<QwenConfig> {
        &self.config
    }
    pub fn pipeline(&self) -> &DecoderPipeline {
        &self.pipeline
    }
}

impl InferenceModel for QwenModel {
    fn device(&self) -> kjarni_transformers::prelude::Device {
        self.pipeline.plan().layers
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.pipeline.context().cloned()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl LanguageModel for QwenModel {
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        _beams: usize,
    ) -> Result<Box<dyn Cache>> {
        let meta = self.config.metadata();
        match self.pipeline.plan().layers {
            kjarni_transformers::prelude::Device::Cpu => {
                let kv_dim = meta.head_dim * meta.num_kv_heads;
                Ok(Box::new(CpuKVCache::new(
                    meta.num_layers,
                    batch_size,
                    max_len,
                    kv_dim,
                )))
            }
            // No GPU cache without a GPU context, which wasm cannot build.
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
            kjarni_transformers::prelude::Device::Wgpu => {
                let ctx = self
                    .context()
                    .ok_or_else(|| anyhow!("GPU context required"))?;
                Ok(Box::new(GpuKVCache::new(
                    &ctx,
                    meta.num_layers,
                    batch_size,
                    meta.num_kv_heads,
                    meta.head_dim,
                    max_len,
                )?))
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
            kjarni_transformers::prelude::Device::Wgpu => {
                Err(anyhow!("GPU cache is not available in WebAssembly"))
            }
        }
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
    fn num_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    // Qwen tokens
    fn bos_token_id(&self) -> Option<u32> {
        Some(self.config.bos_token_id)
    }
    fn eos_token_ids(&self) -> Option<Vec<u32>> {
        Some(self.config.eos_token_id.clone())
    }
    fn eos_token_id(&self) -> Option<u32> {
        self.config.eos_token_id.first().copied()
    }
    fn pad_token_id(&self) -> Option<u32> {
        self.config.pad_token_id
    }
    fn context_size(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn forced_bos_token_id(&self) -> Option<u32> {
        None
    }
    fn forced_eos_token_id(&self) -> Option<u32> {
        None
    }
    fn stop_token_ids(&self) -> std::collections::HashSet<u32> {
        let mut set = std::collections::HashSet::new();
        for id in &self.config.eos_token_id {
            set.insert(*id);
        }

        if let Some(im_end) = self.tokenizer().token_to_id("<|im_end|>") {
            set.insert(im_end);
        }

        set
    }
}

#[async_trait]
impl DecoderLanguageModel for QwenModel {
    fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
        if self.pipeline.cpu_decoder().is_some() {
            Some(self)
        } else {
            None
        }
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
        if self.pipeline.gpu_decoder().is_some() {
            Some(self)
        } else {
            None
        }
    }
    fn autoregressive_loop(&self) -> AutoregressiveLoop {
        AutoregressiveLoop::Pipelined
    }
    fn chat_template(&self) -> Option<&dyn ChatTemplate> {
        self.chat_template.as_deref()
    }
    fn get_default_generation_config(&self) -> GenerationConfig {
        if let Some(d) = &self.generation_defaults {
            return d
                .clone()
                .into_generation_config(self.config.max_position_embeddings);
        }
        GenerationConfig {
            max_new_tokens: Some(512),
            max_length: self.config.max_position_embeddings,
            min_length: 0,
            repetition_penalty: 1.1,
            no_repeat_ngram_size: 0,
            add_bos_token: false,
            strategy: DecodingStrategy::Sample(SamplingParams {
                temperature: 0.7,
                top_k: Some(40),
                top_p: Some(0.8),
                min_p: Some(0.05),
            }),
            speculation: None,
        }
    }
}

impl CpuDecoderOps for QwenModel {
    fn decoder(&self) -> &dyn CpuDecoder {
        self.pipeline.cpu_decoder().unwrap()
    }
    fn project_to_logits(&self, h: &Array3<f32>) -> Result<Array3<f32>> {
        self.pipeline.lm_head().forward_cpu(h)
    }
    fn get_attention_mask(&self, seq: usize, past: usize) -> Result<Array2<f32>> {
        Ok(kjarni_transformers::utils::create_causal_mask(
            seq,
            seq + past,
        ))
    }
    fn embed(&self, tokens: &Array2<u32>, pos: usize) -> Result<Array3<f32>> {
        self.pipeline.embeddings().embed_cpu(tokens, None, pos)
    }
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
impl GpuDecoderOps for QwenModel {
    fn decoder(&self) -> &dyn GpuDecoder {
        self.pipeline.gpu_decoder().unwrap()
    }
    fn get_attention_mask(
        &self,
        ctx: &mut GpuFrameContext,
        seq: usize,
        max: usize,
    ) -> Result<GpuTensor> {
        let mask: Vec<f32> = (0..max).map(|i| if i < seq { 1.0 } else { 0.0 }).collect();

        GpuTensor::create(ctx.context, &mask, vec![1, max], "AttentionMask")
    }
    fn project_to_logits(&self, ctx: &mut GpuFrameContext, h: &GpuTensor) -> Result<GpuTensor> {
        let lm = self.pipeline.lm_head();
        if lm.has_gpu() {
            let (enc, pool) = ctx.resources();
            lm.forward_gpu(enc, pool, h)
        } else {
            // A browser cannot take this path: the readback below completes on the
            // event loop, and `block_on` is what would be blocking it. Hanging the
            // tab is worse than refusing, so refuse and let the caller pick a device.
            #[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
            return Err(anyhow!(
                "the LM head is on the CPU while the decoder is on the GPU, which \
                 needs a blocking GPU readback that a browser cannot perform. Load \
                 the LM head onto the GPU, or run the decoder on the CPU."
            ));

            #[cfg(not(all(target_arch = "wasm32", feature = "wasm-gpu")))]
            pollster::block_on(async {
                let h_cpu = h.to_ndarray_3d().await?;
                let logits = lm.forward_cpu(&h_cpu)?;
                GpuTensor::from_ndarray(ctx.context, &logits)
            })
        }
    }
}
