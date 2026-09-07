use anyhow::{Result, anyhow};
// Filesystem paths are a native-only concern: the wasm builds load from bytes.
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::chat::chatml::ChatMLTemplate;
use crate::chat::llama3::Llama3ChatTemplate;
use crate::chat::mistral::MistralChatTemplate;
use crate::chat::phi3::Phi3ChatTemplate;
use crate::common::HFGenerationDefaults;
use crate::decoder::traits::{CpuDecoder, GpuDecoder};
use crate::loaders::LoadedRoPE;
use crate::models::base::ModelLoadConfig;
#[cfg(not(target_arch = "wasm32"))]
use crate::models::get_default_cache_dir;
use crate::models::{ModelArchitecture, ModelType};
#[cfg(not(target_arch = "wasm32"))]
use crate::models::{download_model_files, registry::WeightsFormat};
use crate::pipeline::DecoderPipeline;
use crate::pipeline::decoder::DecoderPipelineBuilder;
#[cfg(not(target_arch = "wasm32"))]
use crate::tensor::DType;
use crate::traits::{Device, ModelConfig, ModelLayout, ModelMetadata};
use crate::weights::ModelWeights;
use crate::{ChatTemplate, WgpuContext};

pub trait DecoderModelFactory: Sized {
    type Config: ModelConfig + 'static;

    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        rope: &LoadedRoPE,
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(Option<Box<dyn CpuDecoder>>, Option<Box<dyn GpuDecoder>>)>;

    fn new_from_pipeline(
        pipeline: DecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<Self::Config>,
        model_type: Option<ModelType>,
        generation_defaults: Option<HFGenerationDefaults>,
        chat_template: Option<Box<dyn ChatTemplate>>,
    ) -> Self;
    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>>;
}

pub struct DecoderLoader;

impl DecoderLoader {
    /// Fetch from the registry and load. Native only: wasm has no filesystem.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn load_from_registry<M: DecoderModelFactory>(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<M> {
        let info = model_type.info();
        let cache_dir = cache_dir.unwrap_or_else(get_default_cache_dir);
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        let config = load_config.unwrap_or_default();

        let is_quantized_request = matches!(
            config.target_dtype,
            Some(DType::Q4_K) | Some(DType::Q6_K) | Some(DType::Q8_0)
        ) || config.use_gguf;
        let format = if is_quantized_request && info.paths.gguf_url.is_some() {
            log::info!("Configuration requests quantization. Preferring GGUF format.");
            WeightsFormat::GGUF
        } else {
            WeightsFormat::SafeTensors
        };

        download_model_files(&model_dir, &info.paths, format, config.quiet).await?;

        let context = if device.is_gpu() && context.is_none() {
            Some(WgpuContext::new().await?)
        } else {
            context
        };

        Self::load_from_pretrained::<M>(&model_dir, device, context, load_config, Some(model_type))
    }

    /// Load from a directory on disk. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_pretrained<M: DecoderModelFactory>(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<M> {
        log::info!("Loading from {:?}", model_path);
        if let Some(model_type) = model_type {
            log::info!("Model {:?}", model_type.cli_name());
        }
        let weights = ModelWeights::new(model_path)?;
        let load_config: ModelLoadConfig = load_config.unwrap_or_default();

        let config = M::load_config(&weights)?;
        let meta = config.metadata();
        let layout = config.layout();

        // Load Tokenizer
        let tokenizer_path = if model_path.is_file() {
            model_path.parent().unwrap().join("tokenizer.json")
        } else {
            model_path.join("tokenizer.json")
        };
        if !tokenizer_path.exists() {
            return Err(anyhow!("Tokenizer not found at {:?}", tokenizer_path));
        }
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: meta.max_seq_len,
                ..Default::default()
            }))
            .unwrap();
        tokenizer.with_padding(None);

        let rope = LoadedRoPE::new(context.as_ref(), &meta, device.is_gpu())?;

        // try load generation defaults from generation_config.json
        let generation_defaults = Self::try_load_generation_defaults(model_path);

        // Build Backends
        let (cpu_decoder, gpu_decoder) = M::build_backends(
            &weights,
            &meta,
            &layout,
            &rope,
            load_config,
            context.as_ref(),
            device,
        )?;

        //  Build Pipeline via Builder (Coordinates tied weights internally)
        let pipeline: DecoderPipeline = DecoderPipelineBuilder::new(&weights, config.clone())
            .with_load_config(load_config)
            .with_backends(cpu_decoder, gpu_decoder)
            .with_context_opt(context)
            .build()?;
        // Auto-detect template based on ModelType
        let chat_template: Option<Box<dyn ChatTemplate>> = model_type.and_then(|mt| {
            if !mt.is_instruct_model() {
                return None;
            }

            match mt.architecture() {
                ModelArchitecture::Llama => {
                    Some(Box::new(Llama3ChatTemplate::for_generation()) as Box<dyn ChatTemplate>)
                }
                ModelArchitecture::Qwen2 => {
                    Some(Box::new(ChatMLTemplate::new()) as Box<dyn ChatTemplate>)
                }
                ModelArchitecture::Mistral => {
                    Some(Box::new(MistralChatTemplate::new()) as Box<dyn ChatTemplate>)
                }
                ModelArchitecture::Phi3 => {
                    Some(Box::new(Phi3ChatTemplate::new()) as Box<dyn ChatTemplate>)
                }

                // Fallback
                _ => None,
            }
        });
        Ok(M::new_from_pipeline(
            pipeline,
            tokenizer,
            config,
            model_type,
            generation_defaults,
            chat_template,
        ))
    }

    /// Load a decoder straight from an unpacked `.kjq` container.
    ///
    /// `load_from_bytes` takes f32 safetensors, which for a `KJQ8` file would
    /// mean expanding the weights it was written to keep compressed. This keeps
    /// `BlockQ8_0` tensors quantised all the way into `LinearData::Q8_0`, which
    /// is what lets a 494M parameter decoder fit under wasm32's 2GB cap on a
    /// single allocation.
    ///
    /// A `KJQ1` container still works and still dequantises, so the same call
    /// handles both encodings.
    pub fn load_from_kjq<M: DecoderModelFactory>(
        unpacked: &crate::weights::kjq::KjqUnpacked,
        load_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<M> {
        use crate::weights::kjq::KjqEncoding;

        let weights = ModelWeights::from_kjq(unpacked)?;
        let mut load_config: ModelLoadConfig = load_config.unwrap_or_default();

        // The container decides the dtype: a KJQ8 file exists precisely so its
        // weights are not expanded, and honouring a caller's F32 request here
        // would defeat that silently.
        if unpacked.encoding == KjqEncoding::Kjq8 {
            load_config.target_dtype = Some(crate::tensor::DType::Q8_0);
        }

        Self::build_from_parts::<M>(
            weights,
            &unpacked.config_json,
            unpacked.tokenizer_json.as_bytes(),
            load_config,
            model_type,
        )
    }

    /// Load a decoder from raw safetensors plus its config and tokenizer.
    ///
    /// The browser counterpart to `load_from_pretrained`: there is no filesystem to
    /// read from and no GPU context to build, so everything arrives as bytes and the
    /// plan is always CPU. Mirrors `EncoderLoader::load_from_bytes`.
    pub fn load_from_bytes<M: DecoderModelFactory>(
        safetensors_data: &[u8],
        config_json: &str,
        tokenizer_json: &[u8],
        load_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<M> {
        let weights = ModelWeights::from_safetensors_bytes(safetensors_data, config_json)?;
        let load_config: ModelLoadConfig = load_config.unwrap_or_default();
        Self::build_from_parts::<M>(
            weights,
            config_json,
            tokenizer_json,
            load_config,
            model_type,
        )
    }

    /// The half of `load_from_bytes` that does not care where the bytes came from.
    fn build_from_parts<M: DecoderModelFactory>(
        weights: ModelWeights,
        config_json: &str,
        tokenizer_json: &[u8],
        load_config: ModelLoadConfig,
        model_type: Option<ModelType>,
    ) -> Result<M> {
        let _ = config_json;

        let config = M::load_config(&weights)?;
        let meta = config.metadata();
        let layout = config.layout();

        let mut tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: meta.max_seq_len,
            ..Default::default()
        }));
        tokenizer.with_padding(None);

        // The wasm constructor takes no context and no GPU flag.
        #[cfg(not(target_arch = "wasm32"))]
        let rope = LoadedRoPE::new(None, &meta, false)?;
        #[cfg(target_arch = "wasm32")]
        let rope = LoadedRoPE::new(&meta)?;

        let (cpu_decoder, gpu_decoder) = M::build_backends(
            &weights,
            &meta,
            &layout,
            &rope,
            load_config,
            None,
            Device::Cpu,
        )?;

        let pipeline: DecoderPipeline = DecoderPipelineBuilder::new(&weights, config.clone())
            .with_load_config(load_config)
            .with_backends(cpu_decoder, gpu_decoder)
            .build()?;

        let chat_template: Option<Box<dyn ChatTemplate>> = model_type.and_then(|mt| {
            if !mt.is_instruct_model() {
                return None;
            }
            match mt.architecture() {
                ModelArchitecture::Llama => {
                    Some(Box::new(Llama3ChatTemplate::for_generation()) as Box<dyn ChatTemplate>)
                }
                ModelArchitecture::Qwen2 => {
                    Some(Box::new(ChatMLTemplate::new()) as Box<dyn ChatTemplate>)
                }
                ModelArchitecture::Mistral => {
                    Some(Box::new(MistralChatTemplate::new()) as Box<dyn ChatTemplate>)
                }
                ModelArchitecture::Phi3 => {
                    Some(Box::new(Phi3ChatTemplate::new()) as Box<dyn ChatTemplate>)
                }
                _ => None,
            }
        });

        Ok(M::new_from_pipeline(
            pipeline,
            tokenizer,
            config,
            model_type,
            None,
            chat_template,
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn try_load_generation_defaults(model_path: &Path) -> Option<HFGenerationDefaults> {
        let gen_config_path = if model_path.is_file() {
            model_path.parent()?.join("generation_config.json")
        } else {
            model_path.join("generation_config.json")
        };

        let json = std::fs::read_to_string(&gen_config_path).ok()?;
        HFGenerationDefaults::from_json(&json).ok()
    }
}
