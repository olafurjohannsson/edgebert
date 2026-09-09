//! Generic sequence classifier for text classification tasks.
//!
//! Supports BERT, DistilBERT, RoBERTa, and other encoder-only models with
//! classification heads for sentiment, emotion, toxicity, NLI, etc.
//!
//! For BART-based zero-shot classification, use `BartZeroShotClassifier` instead.

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
    activations::softmax_inplace,
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

pub mod configs;

#[cfg(test)]
mod tests;

use crate::models::sequence_classifier::configs::RobertaConfig;
use crate::{BertConfig, DistilBertConfig};

/// A generic sequence classifier for encoder-only models.
pub struct SequenceClassifier {
    pipeline: EncoderPipeline,
    tokenizer: Tokenizer,
    config: Arc<dyn ModelConfig + Send + Sync>,
    model_type: Option<ModelType>,
    labels: Option<Vec<String>>,
}

impl EncoderModelFactory for SequenceClassifier {
    fn load_config(weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>> {
        // Auto-detect config type
        if weights.is_distilbert() {
            Ok(Arc::new(DistilBertConfig::from_json(
                weights.config_json(),
            )?))
        } else if weights.is_roberta() || weights.is_distilroberta() {
            Ok(Arc::new(RobertaConfig::from_json(weights.config_json())?))
        } else {
            // Default to BERT
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
        // Get labels from config if available
        let config: serde_json::Value =
            serde_json::from_str(weights.config_json()).unwrap_or(serde_json::Value::Null);

        let labels = config
            .get("id2label")
            .and_then(|v| v.as_object())
            .map(|obj| {
                let mut labels: Vec<(usize, String)> = obj
                    .iter()
                    .filter_map(|(k, v)| {
                        let idx: usize = k.parse().ok()?;
                        let label = v.as_str()?.to_string();
                        Some((idx, label))
                    })
                    .collect();
                labels.sort_by_key(|(idx, _)| *idx);
                labels.into_iter().map(|(_, label)| label).collect()
            });

        Ok(Some(CpuSequenceClassificationHead::from_weights(
            weights,
            load_config,
            labels,
        )?))
    }

    fn pooling_strategy() -> PoolingStrategy {
        PoolingStrategy::Cls // Classification uses CLS token
    }

    fn new_from_pipeline(
        pipeline: EncoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<dyn ModelConfig>,
        model_type: Option<ModelType>,
    ) -> Self {
        let labels = pipeline.labels().map(|l| l.to_vec());

        Self {
            pipeline,
            tokenizer,
            config,
            model_type,
            labels,
        }
    }
}

impl SequenceClassifier {
    /// Run the encoder on the GPU, or `None` when there is no GPU backend.
    ///
    /// Extracted from the inference path so the wasm build has something to call:
    /// the GPU tensor types do not exist there, so the body cannot simply be
    /// `#[cfg]`-ed out of the middle of an `if let` chain.
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    async fn forward_gpu(
        &self,
        input_ids: &ndarray::Array2<u32>,
        attention_mask: &ndarray::Array2<f32>,
        token_type_ids: &ndarray::Array2<u32>,
    ) -> Result<Option<ndarray::Array3<f32>>> {
        let Some(gpu_encoder) = self.pipeline.gpu_encoder() else {
            return Ok(None);
        };
        let context = self
            .pipeline
            .context()
            .ok_or_else(|| anyhow!("GPU context required for GPU encoder"))?;
        let pool: std::sync::Arc<tokio::sync::Mutex<GpuTensorPool>> = context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(context, pool_guard);
        let (encoder_cmd, pool_ref) = frame.resources();

        let input_ids_gpu = GpuTensor::from_ndarray(context, input_ids)?;
        let attention_mask_gpu = GpuTensor::from_ndarray(context, attention_mask)?;
        let token_types_gpu = GpuTensor::from_ndarray(context, token_type_ids)?;

        #[allow(
            deprecated,
            reason = "internal caller of an API deprecated without a named replacement"
        )]
        let gpu_output = gpu_encoder.forward(
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

    pub fn config(&self) -> &dyn ModelConfig {
        self.config.as_ref()
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

    /// Create classifier from local model directory.
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

    /// Classify text, returning the top prediction.
    pub async fn classify(&self, text: &str) -> Result<ClassificationResult> {
        let results = self.classify_top_k(text, 1).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No results"))
    }

    /// Classify text, returning all scores (after softmax).
    pub async fn classify_scores(&self, text: &str) -> Result<Vec<f32>> {
        let batch_scores = self.classify_scores_batch(&[text]).await?;
        batch_scores
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No results"))
    }

    /// Classify text, returning top-k predictions.
    pub async fn classify_top_k(&self, text: &str, k: usize) -> Result<Vec<ClassificationResult>> {
        let scores = self.classify_scores(text).await?;
        self.scores_to_top_k(&scores, k)
    }

    /// Batch classify multiple texts, returning top-k per text.
    pub async fn classify_batch(
        &self,
        texts: &[&str],
        top_k: usize,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let batch_scores = self.classify_scores_batch(texts).await?;

        let labels = self.labels.as_ref().ok_or_else(|| {
            anyhow!("Model has no label mapping. Use classify_scores_batch() instead.")
        })?;

        let mut all_results = Vec::with_capacity(texts.len());

        for scores in batch_scores {
            // Map to (label, score) tuples
            let mut results: Vec<(String, f32)> = labels
                .iter()
                .zip(scores.iter())
                .map(|(label, &score)| (label.clone(), score))
                .collect();

            // Sort by score descending
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(top_k);
            all_results.push(results);
        }

        Ok(all_results)
    }

    /// Batch classify, returning all scores.
    pub async fn classify_scores_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let logits = self.predict_logits(texts).await?;

        // Apply softmax to each row
        Ok(logits
            .into_iter()
            .map(|mut row| {
                softmax_inplace(&mut row);
                row
            })
            .collect())
    }

    /// Get raw logits (before softmax).
    pub async fn predict_logits(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
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

        let hidden_states = if let Some(hidden) = self
            .forward_gpu(&input_ids, &attention_mask, &token_type_ids)
            .await?
        {
            hidden
        } else {
            let cpu_ops = self
                .encoder_cpu_ops()
                .ok_or_else(|| anyhow!("No CPU encoder available"))?;
            let cpu_encoder = self
                .pipeline
                .cpu_encoder()
                .ok_or_else(|| anyhow!("No CPU encoder available"))?;

            let hidden_states = cpu_ops.embed_tokens(&input_ids, Some(&token_type_ids), 2)?;

            let normalized = cpu_encoder.embed_norm(&hidden_states)?;

            // Forward through CPU encoder
            cpu_encoder
                .forward(&normalized, &attention_mask)?
                .last_hidden_state
        };

        // Forward through head
        let logits = self
            .pipeline
            .cpu_head()
            .ok_or_else(|| anyhow!("No classification head available"))?
            .forward(&hidden_states, Some(&attention_mask))?;

        Ok(logits.outer_iter().map(|row| row.to_vec()).collect())
    }

    #[expect(
        dead_code,
        reason = "not referenced yet; kept until the path that needs it lands"
    )]
    fn scores_to_top_result(&self, scores: &[f32]) -> Result<ClassificationResult> {
        let (idx, &score) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| anyhow!("Empty scores"))?;

        let label = self
            .labels
            .as_ref()
            .and_then(|l| l.get(idx))
            .cloned()
            .unwrap_or_else(|| format!("LABEL_{}", idx));

        Ok(ClassificationResult {
            label,
            score,
            index: idx,
        })
    }

    fn scores_to_top_k(&self, scores: &[f32], k: usize) -> Result<Vec<ClassificationResult>> {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        Ok(indexed
            .into_iter()
            .map(|(idx, score)| {
                let label = self
                    .labels
                    .as_ref()
                    .and_then(|l| l.get(idx))
                    .cloned()
                    .unwrap_or_else(|| format!("LABEL_{}", idx));
                ClassificationResult {
                    label,
                    score,
                    index: idx,
                }
            })
            .collect())
    }

    pub fn pipeline(&self) -> &EncoderPipeline {
        &self.pipeline
    }

    pub fn labels(&self) -> Option<&[String]> {
        self.labels.as_deref()
    }

    pub fn num_labels(&self) -> usize {
        self.pipeline.num_labels().unwrap_or(2)
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

/// Result of a classification.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
    pub index: usize,
}
impl LanguageModel for SequenceClassifier {
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
        Err(anyhow!("Classifiers do not use KV cache"))
    }
}

impl InferenceModel for SequenceClassifier {
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

impl CpuEncoderOps for SequenceClassifier {
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
impl GpuEncoderOps for SequenceClassifier {
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
impl EncoderLanguageModel for SequenceClassifier {
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
