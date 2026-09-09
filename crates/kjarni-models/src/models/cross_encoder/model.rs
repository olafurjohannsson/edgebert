//! Cross-encoder for semantic similarity and reranking.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::gpu::{GpuFrameContext, GpuTensor, GpuTensorPool};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::models::base::ModelInput;
// Filesystem paths are a native-only concern: the wasm builds load from bytes.
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::cpu::encoder::traits::GpuEncoderOps;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::gpu::encoder::GpuTransformerEncoder;
#[cfg(not(target_arch = "wasm32"))]
use kjarni_transformers::pipeline::EncoderLoader;
use kjarni_transformers::{
    WgpuContext,
    cpu::encoder::{
        CpuTransformerEncoder,
        classifier::CpuSequenceClassificationHead,
        config::PoolingStrategy,
        traits::{CpuEncoder, CpuEncoderOps, EncoderLanguageModel, GpuEncoder},
    },
    models::base::ModelLoadConfig,
    models::{LanguageModel, ModelType},
    pipeline::{EncoderModelFactory, EncoderPipeline},
    traits::{Cache, Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

use crate::BertConfig;
use crate::models::sequence_classifier::configs::MiniLMCrossEncoderConfig;

/// Cross-encoder for semantic similarity and reranking.
pub struct CrossEncoder {
    pipeline: EncoderPipeline,
    tokenizer: Tokenizer,
    config: Arc<dyn ModelConfig + Send + Sync>,
    model_type: Option<ModelType>,
}

impl EncoderModelFactory for CrossEncoder {
    fn load_config(weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>> {
        // Cross-encoders are typically BERT-based
        if weights
            .config_json()
            .contains("cross-encoder/ms-marco-MiniLM")
        {
            Ok(Arc::new(MiniLMCrossEncoderConfig::from_json(
                weights.config_json(),
            )?))
        } else {
            // Default to BERT config
            Ok(Arc::new(BertConfig::from_json(weights.config_json())?))
        }
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
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(Option<Box<dyn CpuEncoder>>, Option<Box<dyn GpuEncoder>>)> {
        let mut cpu = None;
        let mut gpu = None;

        match device {
            Device::Cpu => {
                cpu = Some(Box::new(CpuTransformerEncoder::new(
                    weights,
                    meta.clone(),
                    layout.clone(),
                    load_config,
                )?) as Box<dyn CpuEncoder>);
            }
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
            Device::Wgpu => {
                let ctx = context.ok_or_else(|| anyhow!("GPU context required"))?;
                gpu = Some(Box::new(GpuTransformerEncoder::new(
                    weights,
                    ctx.clone(),
                    meta.clone(),
                    layout.clone(),
                    load_config,
                )?) as Box<dyn GpuEncoder>);
            }
            // No GPU backend exists on wasm, and no context can be built,
            // so this arm is unreachable rather than merely unsupported.
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
            Device::Wgpu => {
                return Err(anyhow!("GPU inference is not available in WebAssembly"));
            }
        }

        Ok((cpu, gpu))
    }

    fn build_head(
        weights: &ModelWeights,
        load_config: &ModelLoadConfig,
    ) -> Result<Option<CpuSequenceClassificationHead>> {
        Ok(Some(CpuSequenceClassificationHead::from_weights(
            weights,
            load_config,
            None,
        )?))
    }

    fn pooling_strategy() -> PoolingStrategy {
        PoolingStrategy::Cls
    }

    fn new_from_pipeline(
        pipeline: EncoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<dyn ModelConfig>,
        model_type: Option<ModelType>,
    ) -> Self {
        Self {
            pipeline,
            tokenizer,
            config,
            model_type,
        }
    }
}

impl CrossEncoder {
    /// Run the encoder on the GPU, or `None` when there is no GPU backend.
    ///
    /// Extracted from the inference path so the wasm build has something to call:
    /// the GPU tensor types do not exist there, so the body cannot simply be
    /// `#[cfg]`-ed out of the middle of an `else if` chain.
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    async fn forward_gpu(
        &self,
        input_ids: &ndarray::Array2<u32>,
        attention_mask: &ndarray::Array2<f32>,
        token_type_ids: &ndarray::Array2<u32>,
    ) -> Result<Option<ndarray::Array3<f32>>> {
        let Some(ops) = self.encoder_gpu_ops() else {
            return Ok(None);
        };
        let context = self
            .context()
            .ok_or_else(|| anyhow!("GPU model missing context"))?;
        let pool = context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&context, pool_guard);
        let (encoder_cmd, pool_ref) = frame.resources();

        let input_ids_gpu = GpuTensor::from_ndarray(&context, input_ids)?;
        let attention_mask_gpu = GpuTensor::from_ndarray(&context, attention_mask)?;
        let token_types_gpu = GpuTensor::from_ndarray(&context, token_type_ids)?;

        #[allow(
            deprecated,
            reason = "internal caller of an API deprecated without a named replacement"
        )]
        let gpu_output = ops.encoder().forward(
            encoder_cmd,
            pool_ref,
            ModelInput::TokensGpu(&input_ids_gpu),
            &attention_mask_gpu,
            Some(ModelInput::TokensGpu(&token_types_gpu)),
        )?;

        frame.finish();

        Ok(Some(gpu_output.last_hidden_state.to_ndarray_3d().await?))
    }

    /// No GPU backend exists in WebAssembly, so the CPU path is the only path.
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
    async fn forward_gpu(
        &self,
        _input_ids: &ndarray::Array2<u32>,
        _attention_mask: &ndarray::Array2<f32>,
        _token_type_ids: &ndarray::Array2<u32>,
    ) -> Result<Option<ndarray::Array3<f32>>> {
        Ok(None)
    }

    /// Native only: loading from the registry or from disk needs a filesystem,
    /// which wasm does not have. Browser callers construct from bytes instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        EncoderLoader::load_from_registry::<Self>(
            model_type,
            cache_dir,
            device,
            context,
            load_config,
        )
        .await
    }

    /// Native only: loading from the registry or from disk needs a filesystem,
    /// which wasm does not have. Browser callers construct from bytes instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<Self> {
        EncoderLoader::load_from_pretrained::<Self>(
            model_path,
            device,
            context,
            load_config,
            model_type,
        )
    }
    /// Score a single query-document pair.
    pub async fn predict_pair(&self, query: &str, document: &str) -> Result<f32> {
        let scores = self.predict_pairs(&[(query, document)]).await?;
        scores
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No score returned"))
    }

    /// Score multiple query-document pairs.
    pub async fn predict_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize pairs - the tokenizer handles [CLS] query [SEP] document [SEP]
        let encodings = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        let mut input_ids = ndarray::Array2::<u32>::zeros((batch_size, max_len));
        let mut attention_mask = ndarray::Array2::<f32>::zeros((batch_size, max_len));
        let mut token_type_ids = ndarray::Array2::<u32>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            for (j, &id) in encoding.get_ids().iter().enumerate() {
                input_ids[[i, j]] = id;
                attention_mask[[i, j]] = encoding.get_attention_mask()[j] as f32;
                token_type_ids[[i, j]] = encoding.get_type_ids()[j];
            }
        }

        let hidden_states = if let Some(ops) = self.encoder_cpu_ops() {
            ops.forward_tokens(&input_ids, Some(&attention_mask), Some(&token_type_ids), 0)?
                .last_hidden_state
        } else if let Some(hidden) = self
            .forward_gpu(&input_ids, &attention_mask, &token_type_ids)
            .await?
        {
            hidden
        } else {
            return Err(anyhow!(
                "No available CPU or GPU encoder implementation for this model."
            ));
        };

        // Forward through head // TODO: lm head on GPU
        let logits = self
            .pipeline
            .cpu_head()
            .ok_or_else(|| anyhow!("No classification head available"))?
            .forward(&hidden_states, Some(&attention_mask))?;

        // Extract scores (first column for regression)
        Ok(logits.column(0).to_vec())
    }

    /// Rerank documents by relevance to a query.
    pub async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let pairs: Vec<(&str, &str)> = documents.iter().map(|doc| (query, *doc)).collect();
        let scores = self.predict_pairs(&pairs).await?;

        let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(ranked)
    }

    /// Rerank and return only top-k results.
    pub async fn rerank_top_k(
        &self,
        query: &str,
        documents: &[&str],
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let mut ranked = self.rerank(query, documents).await?;
        ranked.truncate(k);
        Ok(ranked)
    }

    pub fn pipeline(&self) -> &EncoderPipeline {
        &self.pipeline
    }

    pub fn max_seq_length(&self) -> usize {
        self.pipeline.max_seq_length()
    }

    pub fn hidden_size(&self) -> usize {
        self.pipeline.hidden_size()
    }

    pub fn model_type(&self) -> Option<ModelType> {
        self.model_type
    }

    pub fn device(&self) -> Device {
        self.pipeline.plan().layers
    }
}

impl LanguageModel for CrossEncoder {
    fn vocab_size(&self) -> usize {
        self.pipeline.vocab_size()
    }
    fn hidden_size(&self) -> usize {
        self.pipeline.hidden_size()
    }
    fn num_layers(&self) -> usize {
        self.pipeline.num_layers()
    }
    fn num_heads(&self) -> usize {
        self.config.metadata().num_attention_heads
    }
    fn context_size(&self) -> usize {
        self.pipeline.max_seq_length()
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn bos_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("[CLS]")
    }
    fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("[SEP]")
    }
    fn pad_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("[PAD]")
    }
    fn forced_bos_token_id(&self) -> Option<u32> {
        None
    }
    fn forced_eos_token_id(&self) -> Option<u32> {
        None
    }
    fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
        Err(anyhow!("CrossEncoder does not use KV cache"))
    }
}

impl InferenceModel for CrossEncoder {
    fn device(&self) -> Device {
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

impl CpuEncoderOps for CrossEncoder {
    fn encoder(&self) -> &dyn CpuEncoder {
        self.pipeline
            .cpu_encoder()
            .expect("CPU encoder not available")
    }
    fn embed_tokens(
        &self,
        input_ids: &ndarray::Array2<u32>,
        token_type_ids: Option<&ndarray::Array2<u32>>,
        pos: usize,
    ) -> Result<ndarray::Array3<f32>> {
        self.pipeline
            .embeddings()
            .embed_cpu(input_ids, token_type_ids, pos)
    }
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
impl GpuEncoderOps for CrossEncoder {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.pipeline
            .gpu_encoder()
            .expect("GPU encoder not available")
    }
    fn embed_tokens(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input_ids: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
        pos: usize,
    ) -> Result<GpuTensor> {
        self.pipeline()
            .embeddings()
            .embed(cmd_encoder, pool, input_ids, token_type_ids, pos)
    }
}

#[async_trait]
impl EncoderLanguageModel for CrossEncoder {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
        if self.pipeline.cpu_encoder().is_some() {
            Some(self)
        } else {
            None
        }
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
        if self.pipeline.gpu_encoder().is_some() {
            Some(self)
        } else {
            None
        }
    }
}
