//! GPT-2 style decoder-only language model.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::traits::{InferenceModel, ModelConfig};
use ndarray::{Array2, Array3};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use kjarni_transformers::gpu::{GpuFrameContext, GpuTensor, cache::GpuKVCache};
use kjarni_transformers::gpu_ops::primitives::linear::GpuLinearLayer;
use kjarni_transformers::{
    WgpuContext,
    decoder::prelude::*,
    linear_layer::LinearLayer,
    models::{
        LanguageModel, ModelArchitecture, ModelType, base::AutoregressiveLoop, download_model_files,
    },
    prelude::*,
    weights::ModelWeights,
};

use crate::models::gpt2::{
    config::Gpt2Config, cpu_decoder::Gpt2CpuDecoder, gpu_decoder::Gpt2GpuDecoder,
};
use kjarni_transformers::models::get_default_cache_dir;

pub struct Gpt2Model {
    cpu_decoder: Option<Gpt2CpuDecoder>,
    gpu_decoder: Option<Gpt2GpuDecoder>,
    tokenizer: Tokenizer,
    config: Arc<Gpt2Config>,
    lm_head: LinearLayer,
    gpu_lm_head_transposed: Option<GpuTensor>,
    gpu_lm_head_layer: Option<GpuLinearLayer>,
    device: Device,
    context: Option<Arc<WgpuContext>>,
}

impl Gpt2Model {
    /// A list of the specific model types supported by this implementation.
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::DistilGpt2,
        ModelType::Gpt2,
        // ModelType::Gpt2Medium,
        // ModelType::Gpt2Large,
        // ModelType::Gpt2XL,
    ];
    pub fn concrete_config(&self) -> &Arc<Gpt2Config> {
        &self.config
    }

    /// Creates a `Gpt2Model` from the HuggingFace model registry.
    ///
    /// This will download the model files to a local cache directory if they
    /// are not already present.
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Unsupported GPT-2 model type: {:?}", model_type));
        }
        // if model_type.info().architecture != ModelArchitecture::Decoder {
        //     return Err(anyhow!("Model {:?} is not a decoder model.", model_type));
        // }
        let info = model_type.info();

        match info.architecture {
            ModelArchitecture::GPT => {}
            _ => {
                return Err(anyhow!(
                    "Model {:?} is not an GPT (architecture: {:?})",
                    model_type,
                    info.architecture
                ));
            }
        }

        let cache_dir = cache_dir.unwrap_or_else(get_default_cache_dir);
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        download_model_files(
            &model_dir,
            &model_type.info().paths,
            kjarni_transformers::models::registry::WeightsFormat::SafeTensors,
            true,
        )
        .await?;
        Self::from_pretrained(&model_dir, model_type, device, context, decoder_config)
    }

    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let tokenizer =
            Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| anyhow!(e))?;

        let config: Arc<Gpt2Config> = {
            let mut cfg = serde_json::from_str::<Gpt2Config>(weights.config_json())?;
            if model_type == ModelType::DistilGpt2 {
                // Special handling for DistilGPT2's unique weight naming convention.
                cfg.set_model_type("distilgpt2".to_string());
            };
            Arc::new(cfg)
        };

        let meta = config.metadata();
        let layout = config.layout();

        let lm_head = LinearLayer::builder(&weights, &layout.lm_head)
            .with_target_dtype(decoder_config.unwrap_or_default().target_dtype)
            .with_optional_bias(None)
            .build()?;

        assert_eq!(
            lm_head.shape(),
            [config.vocab_size, meta.hidden_size],
            "LM head shape mismatch"
        );

        // Split CPU/GPU paths
        match device {
            Device::Cpu => {
                log::debug!("Building CPU decoder...");
                let cpu_decoder = Gpt2CpuDecoder::new(&weights, config.clone())?;

                Ok(Self {
                    cpu_decoder: Some(cpu_decoder),
                    gpu_decoder: None,
                    tokenizer,
                    config,
                    lm_head,
                    gpu_lm_head_transposed: None,
                    gpu_lm_head_layer: None,
                    device,
                    context: None,
                })
            }
            Device::Wgpu => {
                log::debug!("Building GPU decoder...");
                let ctx = context.ok_or_else(|| anyhow!("GPU device requires context"))?;

                ctx.print_memory_usage();

                let gpu_decoder = Gpt2GpuDecoder::new(
                    &ctx,
                    &weights,
                    config.clone(),
                    decoder_config.unwrap_or_default(),
                )?;

                let gpu_lm_head = lm_head.to_gpu(&ctx)?;

                log::debug!("✓ GPU model loaded successfully");

                Ok(Self {
                    cpu_decoder: None,
                    gpu_decoder: Some(gpu_decoder),
                    tokenizer,
                    config,
                    lm_head,
                    gpu_lm_head_transposed: Some(gpu_lm_head),
                    gpu_lm_head_layer: Some(GpuLinearLayer::new(&ctx)),
                    device,
                    context: Some(ctx),
                })
            }
        }
    }
}

impl InferenceModel for Gpt2Model {
    fn device(&self) -> Device {
        self.device
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.context.clone()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl LanguageModel for Gpt2Model {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn context_size(&self) -> usize {
        self.config.metadata().max_seq_len
    }
    fn forced_bos_token_id(&self) -> Option<u32> {
        None
    }
    fn forced_eos_token_id(&self) -> Option<u32> {
        None
    }
    fn vocab_size(&self) -> usize {
        self.config.metadata().vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.config.metadata().hidden_size
    }
    fn num_heads(&self) -> usize {
        self.config.metadata().num_attention_heads
    }
    fn num_layers(&self) -> usize {
        self.config.metadata().num_layers
    }
    fn eos_token_id(&self) -> Option<u32> {
        Some(50256)
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(50256)
    }
    fn pad_token_id(&self) -> Option<u32> {
        Some(50256)
    }
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        _num_beams: usize,
    ) -> Result<Box<dyn Cache>> {
        Ok(match self.device() {
            Device::Cpu => Box::new(CpuKVCache::new(
                self.num_layers(),
                batch_size,
                max_len,
                self.hidden_size(),
            )),
            Device::Wgpu => {
                let context = self
                    .context()
                    .ok_or_else(|| anyhow!("GPU model missing context"))?;
                let head_dim = self.hidden_size() / self.num_heads();
                Box::new(GpuKVCache::new(
                    &context,
                    self.num_layers(),
                    batch_size,
                    self.num_heads(),
                    head_dim,
                    max_len,
                )?)
            }
        })
    }
}

impl CpuDecoderOps for Gpt2Model {
    fn decoder(&self) -> &dyn CpuDecoder {
        self.cpu_decoder
            .as_ref()
            .expect("CPU decoder not initialized - use Device::Cpu")
    }

    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden) = hidden_states.dim();
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch * seq, hidden))?;
        let logits_2d = self.lm_head.matmul(&hidden_2d);

        logits_2d
            .into_shape_with_order((batch, seq, self.vocab_size()))
            .map_err(|e| anyhow!(e))
    }

    fn get_attention_mask(&self, seq_len: usize, past_len: usize) -> Result<Array2<f32>> {
        let total_len = seq_len + past_len;
        Ok(Array2::ones((1, total_len)))
    }

    fn embed(&self, tokens: &Array2<u32>, pos: usize) -> Result<Array3<f32>> {
        let decoder: &Gpt2CpuDecoder = self
            .cpu_decoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No GPT2 CPU Decoder"))?;
        let tokens = decoder.embeddings.forward(tokens, None, pos, false);
        Ok(tokens)
    }
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
impl GpuDecoderOps for Gpt2Model {
    fn decoder(&self) -> &dyn GpuDecoder {
        self.gpu_decoder
            .as_ref()
            .expect("GPU decoder not initialized - use Device::Wgpu")
    }

    fn get_attention_mask(
        &self,
        ctx: &mut GpuFrameContext,
        seq_len: usize,
        max_len: usize,
    ) -> Result<GpuTensor> {
        let mask_data: Vec<f32> = (0..max_len)
            .map(|i| if i < seq_len { 1.0 } else { 0.0 })
            .collect();
        GpuTensor::create(ctx.context, &mask_data, vec![1, max_len], "AttentionMask")
    }

    fn project_to_logits(
        &self,
        ctx: &mut GpuFrameContext,
        last_hidden_state: &GpuTensor,
    ) -> Result<GpuTensor> {
        let lm_head = self.gpu_lm_head_transposed.as_ref().unwrap();
        let linear_layer = self.gpu_lm_head_layer.as_ref().unwrap();

        let (batch, seq, _hidden) = last_hidden_state.dims3();
        let vocab_size = lm_head.shape()[0];

        let logits = ctx.pool_guard.get(vec![batch, seq, vocab_size]);

        linear_layer.encode(
            ctx.encoder.as_mut().unwrap(),
            last_hidden_state,
            lm_head,
            &logits,
        );

        Ok(logits)
    }
}

#[async_trait]
impl DecoderLanguageModel for Gpt2Model {
    fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
        if self.device == Device::Cpu {
            Some(self)
        } else {
            None
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
        if self.device == Device::Wgpu {
            Some(self)
        } else {
            None
        }
    }

    fn autoregressive_loop(&self) -> AutoregressiveLoop {
        AutoregressiveLoop::Legacy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};

    use kjarni_transformers::prelude::LanguageModel;

    /// Helper function to load the DistilGPT2 model for testing.
    async fn load_distilgpt2_for_test() -> Result<Gpt2Model> {
        Gpt2Model::from_registry(ModelType::DistilGpt2, None, Device::Cpu, None, None).await
    }

    #[tokio::test]
    #[ignore = "DistilGPT not loaded in CI"]
    async fn test_distilgpt2_architectural_properties() -> Result<()> {
        {
            let model = load_distilgpt2_for_test().await?;
            let config = model.concrete_config();

            assert_eq!(config.vocab_size, 50257);
            assert_eq!(config.n_embd, 768); // hidden size
            assert_eq!(config.n_layer, 6);
            assert_eq!(config.n_head, 12);
            assert_eq!(config.n_ctx, 1024); // max_position_embeddings

            assert_eq!(model.vocab_size(), 50257);
            assert_eq!(model.hidden_size(), 768);
            assert_eq!(model.num_layers(), 6);
            assert_eq!(model.num_heads(), 12);
            assert_eq!(model.max_length(), 1024);

            assert_eq!(model.bos_token_id(), Some(50256));
            assert_eq!(model.eos_token_id(), Some(50256));
            assert_eq!(model.pad_token_id(), Some(50256));
        }
        kjarni_transformers::weights::clear_mmap_cache();
        Ok(())
    }

    #[tokio::test]
    #[ignore = "DistilGPT not loaded in CI"]
    async fn test_distilgpt2_generation_parity_2() -> Result<()> {
        let model_type = ModelType::DistilGpt2;
        let prompt = "The field of Artificial Intelligence has seen a lot of progress";

        let expected_output = "The field of Artificial Intelligence has seen a lot of progress in the past few years, but it is still not clear how much improvement will be made.";

        let config = GenerationConfig {
            max_new_tokens: Some(20),
            strategy: DecodingStrategy::Greedy,
            repetition_penalty: 1.1,
            add_bos_token: false,
            ..Default::default()
        };
        {
            let gpt2_model =
                Gpt2Model::from_registry(model_type, None, Device::Cpu, None, None).await?;

            let generator = DecoderGenerator::new(Arc::new(gpt2_model))?;

            let generated_text = generator.generate(prompt, &config, None).await?;
            let concat_prompt = prompt.to_string() + "" + &generated_text;
            assert_eq!(concat_prompt.trim(), expected_output.trim());
        }
        kjarni_transformers::weights::clear_mmap_cache();
        Ok(())
    }

    #[ignore = "Requires GPU"]
    #[tokio::test]
    async fn test_distilgpt2_generation_parity_cpu_gpu() -> Result<()> {
        let model_type = ModelType::DistilGpt2;
        let prompt = "The field of Artificial Intelligence has seen a lot of progress";

        let config = GenerationConfig {
            max_new_tokens: Some(5),
            strategy: DecodingStrategy::Greedy,
            repetition_penalty: 1.1,
            add_bos_token: false,
            ..Default::default()
        };

        {
            let gpt2_model =
                Gpt2Model::from_registry(model_type, None, Device::Cpu, None, None).await?;

            let generator = DecoderGenerator::new(Arc::new(gpt2_model))?;

            let generated_text = generator.generate(prompt, &config, None).await?;

            let ctx = WgpuContext::new().await?;
            let gpt2_model_2 =
                Gpt2Model::from_registry(model_type, None, Device::Wgpu, Some(ctx), None).await?;
            let generator_2 = DecoderGenerator::new(Arc::new(gpt2_model_2))?;
            let generated_text_2 = generator_2.generate(prompt, &config, None).await?;
            assert_eq!(generated_text.trim(), generated_text_2.trim());
        }
        kjarni_transformers::weights::clear_mmap_cache();
        Ok(())
    }
}
