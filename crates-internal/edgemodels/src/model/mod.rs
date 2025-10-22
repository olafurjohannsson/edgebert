//! GPT model implementations

pub mod bart;
pub mod gptbase;
pub mod distilgpt2;
pub mod gpt2;

use anyhow::Result;
use std::path::PathBuf;

use edgetransformers::wgpu_context::WgpuContext;

pub use bart::BartModel;
pub use gptbase::GPTBase;
pub use distilgpt2::DistilGPT2;
pub use gpt2::GPT2;

use crate::config::BartConfig;
use crate::generation::{generate_encoder_decoder, generate_text};

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    DistilGPT2,
    GPT2,
    GPT2Medium,
    DistilBartCnn12_6,
}

/// Main Generative model for text generation
pub enum GenerativeModel {
    DistilGPT2(DistilGPT2),
    GPT2(GPT2),
    Bart(BartModel),
}

impl GenerativeModel {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained(model_type: ModelType) -> Result<Self> {
        use crate::config::GPTConfig;
        use crate::weights::ModelWeights;

        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("edgegpt");

        std::fs::create_dir_all(&cache_dir)?;

        match model_type {
            ModelType::DistilGPT2 | ModelType::GPT2 | ModelType::GPT2Medium => {
                let model_path = match model_type {
                    ModelType::DistilGPT2 => cache_dir.join("distilgpt2"),
                    ModelType::GPT2 => cache_dir.join("gpt2"),
                    ModelType::GPT2Medium => cache_dir.join("gpt2-medium"),
                    _ => unreachable!(),
                };
                ensure_model_files(model_type, &model_path)?;

                let weights = ModelWeights::load(&model_path)?;
                let config: GPTConfig = serde_json::from_str(&weights.config_json)?;
                let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

                match model_type {
                    ModelType::DistilGPT2 => Ok(GenerativeModel::DistilGPT2(
                        DistilGPT2::from_weights(weights, tokenizer, config)?,
                    )),
                    _ => Ok(GenerativeModel::GPT2(GPT2::from_weights(
                        weights, tokenizer, config,
                    )?)),
                }
            }
            ModelType::DistilBartCnn12_6 => {
                let model_path = cache_dir.join("distilbart-cnn-12-6");
                ensure_model_files(model_type, &model_path)?;

                let weights = ModelWeights::load(&model_path)?;
                let config: BartConfig = serde_json::from_str(&weights.config_json)?;
                let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

                let bart_model = BartModel::from_weights(&weights, config, tokenizer)?;
                Ok(GenerativeModel::Bart(bart_model))
            }
        }
    }

    pub async fn generate1(
        &self,
        prompt: &str,
        config: &crate::generation::GenerationConfig,
        context: &WgpuContext
    ) -> Result<String> {
        match self {
            GenerativeModel::DistilGPT2(model) => {
                unimplemented!("Not implemented");
            }
            GenerativeModel::GPT2(model) => {
                unimplemented!("Not implemented");
            }
            GenerativeModel::Bart(model) => {
                generate_encoder_decoder(model, &model.tokenizer, prompt, config, context).await
            }
        }
    }

    pub fn generate(
        &self,
        prompt: &str,
        config: &crate::generation::GenerationConfig,
    ) -> Result<String> {
        match self {
            GenerativeModel::DistilGPT2(model) => {
                generate_text(&model.base, &model.tokenizer, prompt, config)
            }
            GenerativeModel::GPT2(model) => {
                generate_text(&model.base, &model.tokenizer, prompt, config)
            }
            GenerativeModel::Bart(model) => {
                unimplemented!("Not implemented");
                // generate_encoder_decoder(model, &model.tokenizer, prompt, config)
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_model_files(model_type: ModelType, model_path: &std::path::Path) -> Result<()> {
    let (weights_url, tokenizer_url, config_url) = match model_type {
        ModelType::DistilGPT2 => (
            "https://huggingface.co/distilbert/distilgpt2/resolve/main/model.safetensors",
            "https://huggingface.co/distilbert/distilgpt2/resolve/main/tokenizer.json",
            "https://huggingface.co/distilbert/distilgpt2/resolve/main/config.json",
        ),
        ModelType::GPT2 => (
            "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
            "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json",
            "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
        ),
        ModelType::GPT2Medium => (
            "https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors",
            "https://huggingface.co/openai-community/gpt2-medium/resolve/main/tokenizer.json",
            "https://huggingface.co/openai-community/gpt2-medium/resolve/main/config.json",
        ),
        ModelType::DistilBartCnn12_6 => (
            "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/model.safetensors",
            "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/tokenizer.json",
            "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/config.json",
        ),
    };

    std::fs::create_dir_all(model_path)?;

    let download = |url: &str, path: &std::path::Path| -> anyhow::Result<()> {
        if !path.exists() {
            let resp = reqwest::blocking::get(url)?;
            if !resp.status().is_success() {
                anyhow::bail!("Failed to download {}: {}", url, resp.status());
            }
            let bytes = resp.bytes()?;
            std::fs::write(path, &bytes)?;
        }
        Ok(())
    };

    download(weights_url, &model_path.join("model.safetensors"))?;
    download(tokenizer_url, &model_path.join("tokenizer.json"))?;
    download(config_url, &model_path.join("config.json"))?;

    Ok(())
}
