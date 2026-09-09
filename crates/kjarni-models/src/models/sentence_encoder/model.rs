//! Sentence encoder for semantic similarity and embeddings.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::gpu::{GpuTensor, GpuTensorPool};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::models::base::ModelInput;
// Filesystem paths are a native-only concern: the wasm builds load from bytes.
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

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
        config::{EncodingConfig, PoolingStrategy},
        traits::{
            CpuEncoder, CpuEncoderOps, EncoderLanguageModel, GpuEncoder, SentenceEncoderModel,
        },
    },
    models::base::ModelLoadConfig,
    models::{LanguageModel, ModelType},
    pipeline::{EncoderModelFactory, EncoderPipeline},
    traits::{Cache, Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

use super::configs::{BertConfig, DistilBertConfig};
use crate::MpnetConfig;
use crate::models::sequence_classifier::configs::RobertaConfig;

/// Sentence encoder for semantic similarity tasks.
pub struct SentenceEncoder {
    pipeline: EncoderPipeline,
    tokenizer: Tokenizer,
    config: Arc<dyn ModelConfig + Send + Sync>,
    model_type: Option<ModelType>,
}

impl EncoderModelFactory for SentenceEncoder {
    fn load_config(weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>> {
        if weights.is_distilbert() {
            Ok(Arc::new(DistilBertConfig::from_json(
                weights.config_json(),
            )?))
        } else if weights.is_roberta() || weights.is_distilroberta() {
            Ok(Arc::new(RobertaConfig::from_json(weights.config_json())?))
        } else if weights.is_mpnet() {
            Ok(Arc::new(MpnetConfig::from_json(weights.config_json())?))
        } else {
            // Default to BertConfig for BERT-like models
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

    fn pooling_strategy() -> PoolingStrategy {
        PoolingStrategy::Mean
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

// Public API

impl SentenceEncoder {
    /// Create encoder from HuggingFace model registry.
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

    /// Create encoder from local model directory.
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

    /// Encode text with default mean pooling and normalization.
    pub async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let config = EncodingConfig {
            normalize: true,
            pooling_strategy: self.pipeline.pooling_strategy(),
        };
        <Self as SentenceEncoderModel>::encode(self, text, &config).await
    }

    /// Encode text without normalization (raw embeddings).
    pub async fn encode_raw(&self, text: &str) -> Result<Vec<f32>> {
        let config = EncodingConfig {
            normalize: false,
            pooling_strategy: self.pipeline.pooling_strategy(),
        };
        <Self as SentenceEncoderModel>::encode(self, text, &config).await
    }

    /// Encode with custom pooling strategy and optional normalization.
    pub async fn encode_with(
        &self,
        text: &str,
        pooling_strategy: Option<&str>,
        normalize: bool,
    ) -> Result<Vec<f32>> {
        let strategy = Self::parse_pooling_strategy(pooling_strategy)?;
        let config = EncodingConfig {
            normalize,
            pooling_strategy: strategy,
        };
        <Self as SentenceEncoderModel>::encode(self, text, &config).await
    }

    /// Encode batch of texts with default mean pooling and normalization.
    pub async fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let config = EncodingConfig {
            normalize: true,
            pooling_strategy: self.pipeline.pooling_strategy(),
        };
        <Self as SentenceEncoderModel>::encode_batch(self, texts, &config).await
    }

    /// Encode batch of texts without normalization (raw embeddings).
    pub async fn encode_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let config = EncodingConfig {
            normalize: false,
            pooling_strategy: self.pipeline.pooling_strategy(),
        };
        <Self as SentenceEncoderModel>::encode_batch(self, texts, &config).await
    }

    pub async fn encode_batch_flat(&self, texts: &[&str]) -> Result<(Vec<f32>, usize, usize)> {
        let (hidden_states, attention_mask) = self.get_hidden_states_batch(texts).await?;

        let mut pooled = kjarni_transformers::mean_pool(&hidden_states, &attention_mask)?;

        let _config = EncodingConfig {
            normalize: false,
            pooling_strategy: self.pipeline.pooling_strategy(),
        };

        kjarni_transformers::cpu::encoder::traits::l2_normalize_inplace(&mut pooled);

        let (rows, cols) = pooled.dim();
        let standard_layout = pooled.as_standard_layout();
        let (flat_vec, _) = standard_layout.into_owned().into_raw_vec_and_offset();

        Ok((flat_vec, rows, cols))
    }

    /// Encode batch with custom pooling strategy and optional normalization.
    ///
    /// # Arguments
    /// * `texts` - The texts to encode
    /// * `pooling_strategy` - Pooling method: "mean", "cls", "max", or "lastToken"
    /// * `normalize` - Whether to L2-normalize the output embeddings
    ///
    /// # Example
    /// ```ignore
    /// let mean_raw = encoder.encode_batch_with(&texts, Some("mean"), false).await?;
    /// let cls_norm = encoder.encode_batch_with(&texts, Some("cls"), true).await?;
    /// ```
    pub async fn encode_batch_with(
        &self,
        texts: &[&str],
        pooling_strategy: Option<&str>,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let strategy = Self::parse_pooling_strategy(pooling_strategy)?;
        let config = EncodingConfig {
            normalize,
            pooling_strategy: strategy,
        };
        <Self as SentenceEncoderModel>::encode_batch(self, texts, &config).await
    }

    /// Parse a string pooling strategy into the enum.
    fn parse_pooling_strategy(strategy: Option<&str>) -> Result<PoolingStrategy> {
        match strategy.unwrap_or("mean") {
            "mean" => Ok(PoolingStrategy::Mean),
            "cls" => Ok(PoolingStrategy::Cls),
            "max" => Ok(PoolingStrategy::Max),
            "lastToken" | "last_token" | "last" => Ok(PoolingStrategy::LastToken),
            other => Err(anyhow!(
                "Unknown pooling strategy '{}'. Supported: mean, cls, max, lastToken",
                other
            )),
        }
    }

    pub fn pipeline(&self) -> &EncoderPipeline {
        &self.pipeline
    }

    pub fn embedding_dim(&self) -> usize {
        self.pipeline.hidden_size()
    }

    pub fn max_seq_length(&self) -> usize {
        self.pipeline.max_seq_length()
    }

    pub fn model_type(&self) -> Option<ModelType> {
        self.model_type
    }

    pub fn hidden_size(&self) -> usize {
        self.pipeline.hidden_size()
    }

    pub fn device(&self) -> Device {
        self.pipeline.plan().layers
    }
}

impl LanguageModel for SentenceEncoder {
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
        Err(anyhow!("Sentence encoders do not use KV cache"))
    }
}

impl InferenceModel for SentenceEncoder {
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

impl CpuEncoderOps for SentenceEncoder {
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
impl GpuEncoderOps for SentenceEncoder {
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
impl EncoderLanguageModel for SentenceEncoder {
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
