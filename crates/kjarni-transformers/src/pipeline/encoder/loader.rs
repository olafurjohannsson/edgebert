use std::sync::Arc;
// Filesystem paths are a native-only concern: the wasm builds load from bytes.
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

#[cfg(not(target_arch = "wasm32"))]
use crate::models::get_default_cache_dir;
use crate::{
    Device, ModelType, WgpuContext,
    cpu::encoder::{
        classifier::CpuSequenceClassificationHead,
        config::PoolingStrategy,
        traits::{CpuEncoder, GpuEncoder},
    },
    models::base::ModelLoadConfig,
    pipeline::encoder::{EncoderPipeline, EncoderPipelineBuilder},
    traits::{ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};
use anyhow::{Result, anyhow};

/// Factory trait for encoder-based models.
pub trait EncoderModelFactory: Sized {
    /// Load the model config from weights
    fn load_config(weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>>;

    /// Build the encoder backend(s).
    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(Option<Box<dyn CpuEncoder>>, Option<Box<dyn GpuEncoder>>)>;

    /// Build the classification head
    fn build_head(
        _weights: &ModelWeights,
        _load_config: &ModelLoadConfig,
    ) -> Result<Option<CpuSequenceClassificationHead>> {
        Ok(None)
    }

    /// Get the pooling strategy for this model type.
    fn pooling_strategy() -> PoolingStrategy {
        PoolingStrategy::Mean
    }

    /// Construct the model from a loaded pipeline.
    fn new_from_pipeline(
        pipeline: EncoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<dyn ModelConfig>,
        model_type: Option<ModelType>,
    ) -> Self;
}

pub struct EncoderLoader;

#[cfg(not(target_arch = "wasm32"))]
use crate::models::{download_model_files, registry::WeightsFormat};

/// The sequence length a sentence-transformers model was actually tuned for.
///
/// Three different numbers claim to be this model's context, and they disagree.
/// For `all-MiniLM-L6-v2` the shipped `tokenizer.json` says 128, the
/// `sentence_bert_config.json` that sentence-transformers reads says 256, and
/// `max_position_embeddings` in `config.json` says 512.
///
/// The reference implementation uses `sentence_bert_config.json`, so text past
/// that point is never seen by the encoder there. Taking `max_position_embeddings`
/// instead silently feeds the model positions its sentence-embedding fine-tune
/// never trained, and produces embeddings that disagree with the reference for
/// exactly the long inputs where the difference matters.
///
/// Returns `None` when the model ships no such file, which is the case for
/// anything that is not a sentence-transformers export.
#[allow(dead_code, reason = "reachable only on some targets")]
fn sentence_transformers_max_seq_len(model_path: &std::path::Path) -> Option<usize> {
    #[derive(serde::Deserialize)]
    struct SentenceBertConfig {
        max_seq_length: Option<usize>,
    }

    let raw = std::fs::read_to_string(model_path.join("sentence_bert_config.json")).ok()?;
    let parsed: SentenceBertConfig = serde_json::from_str(&raw).ok()?;
    parsed.max_seq_length.filter(|n| *n > 0)
}

/// The tuned sequence length carried inside a packed config, if present.
///
/// `load_from_bytes` serves the browser, which has no filesystem and therefore no
/// `sentence_bert_config.json` to read. `scripts/quantize_model.py` folds that
/// file's `max_seq_length` into the config it packs into a `.kjq` precisely so this
/// path can find it, and so a model truncates identically in the browser and
/// natively. Older `.kjq` files predate that and simply lack the key.
fn packed_max_seq_len(config_json: &str) -> Option<usize> {
    #[derive(serde::Deserialize)]
    struct Packed {
        max_seq_length: Option<usize>,
    }
    let parsed: Packed = serde_json::from_str(config_json).ok()?;
    parsed.max_seq_length.filter(|n| *n > 0)
}

/// Resolves the truncation length, most specific source first.
///
/// `ceiling` is what the model architecture can actually represent, which for a
/// model with learned position embeddings is the height of that table. Going past
/// it does not fail: `CpuEmbeddings::forward` clamps the slice it adds, so tokens
/// beyond the table keep their word embedding and enter attention with no
/// positional information at all. The result is a plausible-looking vector that is
/// quietly wrong, which is the worst way for this to go.
///
/// So an explicit override past the ceiling is rejected rather than honoured. The
/// caller asked for something the model cannot do, and a load error names the
/// mistake where it was made. A *derived* value past the ceiling is clamped with a
/// warning instead, because a malformed config should not make a model unloadable.
fn resolve_max_seq_len(
    explicit: Option<usize>,
    from_sentence_transformers: Option<usize>,
    from_model_config: usize,
    ceiling: usize,
) -> Result<usize> {
    if let Some(n) = explicit.filter(|n| *n > 0) {
        if ceiling > 0 && n > ceiling {
            return Err(anyhow!(
                "max_sequence_length {n} exceeds what this model can represent ({ceiling}). \
                 Tokens past {ceiling} would receive no position embedding and the \
                 resulting vector would be wrong without any error. Use {ceiling} or less."
            ));
        }
        return Ok(n);
    }

    let derived = match from_sentence_transformers {
        Some(n) if n != from_model_config => {
            log::debug!(
                "using max_seq_length {n} from sentence_bert_config.json \
                 rather than {from_model_config} from the model config"
            );
            n
        }
        Some(n) => n,
        None => from_model_config,
    };

    if ceiling > 0 && derived > ceiling {
        log::warn!(
            "config declares a sequence length of {derived} but the model can only \
             represent {ceiling}; clamping. Inputs longer than {ceiling} are truncated."
        );
        return Ok(ceiling);
    }
    Ok(derived)
}

impl EncoderLoader {
    /// Fetch from the registry and load. Native only: wasm has no filesystem to
    /// cache into, so the browser path is [`EncoderLoader::load_from_bytes`].
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn load_from_registry<M: EncoderModelFactory>(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<M> {
        let info = model_type.info();
        let cache_dir = cache_dir.unwrap_or_else(get_default_cache_dir);
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        // Only silent if the caller asked for it. Downloading hundreds of
        // megabytes with no output reads as a hang.
        let quiet = load_config.as_ref().is_some_and(|c| c.quiet);
        download_model_files(&model_dir, &info.paths, WeightsFormat::SafeTensors, quiet).await?;

        // Create GPU context if needed
        let context = if device.is_gpu() && context.is_none() {
            Some(WgpuContext::new().await?)
        } else {
            context
        };

        Self::load_from_pretrained::<M>(&model_dir, device, context, load_config, Some(model_type))
    }

    /// Load from a directory on disk. Native only, for the same reason as
    /// [`EncoderLoader::load_from_registry`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_pretrained<M: EncoderModelFactory>(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<M> {
        log::info!("Loading encoder from {:?}", model_path);

        let weights = ModelWeights::new(model_path)?;
        let load_config = load_config.unwrap_or_default();

        // Load model-specific config
        let config = M::load_config(&weights)?;
        let meta = config.metadata();
        let layout = config.layout();

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(anyhow!("Tokenizer not found at {:?}", tokenizer_path));
        }
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Configure truncation and padding
        let max_seq_len = resolve_max_seq_len(
            load_config.max_sequence_length,
            sentence_transformers_max_seq_len(model_path),
            meta.max_seq_len,
            meta.max_seq_len,
        )?;
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: max_seq_len,
            ..Default::default()
        }));
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        // Build backends
        let (cpu_encoder, gpu_encoder) = M::build_backends(
            &weights,
            &meta,
            &layout,
            load_config,
            context.as_ref(),
            device,
        )?;

        // Build head (if any)
        let cpu_head = M::build_head(&weights, &load_config)?;

        // Build pipeline
        let pipeline = EncoderPipelineBuilder::new(&weights, config.clone())
            .with_load_config(load_config)
            .with_backends(cpu_encoder, gpu_encoder)
            .with_head(cpu_head)
            .with_pooling_strategy(M::pooling_strategy())
            .with_context(context)
            .build()?;

        // Construct the model
        Ok(M::new_from_pipeline(
            pipeline, tokenizer, config, model_type,
        ))
    }

    /// Load from raw bytes (for WASM)
    pub fn load_from_bytes<M: EncoderModelFactory>(
        safetensors_data: &[u8],
        config_json: &str,
        tokenizer_json: &[u8],
        _device: Device,
        load_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<M> {
        let weights = ModelWeights::from_safetensors_bytes(safetensors_data, config_json)?;
        let load_config = load_config.unwrap_or_default();

        let config = M::load_config(&weights)?;
        let meta = config.metadata();
        let layout = config.layout();

        let mut tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // No filesystem here, so the tuned length has to arrive inside the packed
        // config rather than beside it. See `packed_max_seq_len`.
        let max_seq_len = resolve_max_seq_len(
            load_config.max_sequence_length,
            packed_max_seq_len(config_json),
            meta.max_seq_len,
            meta.max_seq_len,
        )?;
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: max_seq_len,
            ..Default::default()
        }));
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        // WASM is always CPU, no GPU context
        let (cpu_encoder, gpu_encoder) =
            M::build_backends(&weights, &meta, &layout, load_config, None, Device::Cpu)?;

        let cpu_head = M::build_head(&weights, &load_config)?;

        let pipeline = EncoderPipelineBuilder::new(&weights, config.clone())
            .with_load_config(load_config)
            .with_backends(cpu_encoder, gpu_encoder)
            .with_head(cpu_head)
            .with_pooling_strategy(M::pooling_strategy())
            .build()?;

        Ok(M::new_from_pipeline(
            pipeline, tokenizer, config, model_type,
        ))
    }
}
