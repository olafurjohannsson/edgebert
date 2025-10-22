//! BERT model implementations

use anyhow::Result;
use std::path::PathBuf;

pub use crate::model::bertbase::BertBase;
pub use crate::model::bi_encoder::BertBiEncoder;
pub use crate::model::cross_encoder::BertCrossEncoder;

#[derive(Debug, Clone, Copy)]
pub enum BertModelType {
    MiniLML6V2BiEncoder,
    MiniLML6V2CrossEncoder,
}

/// Main BERT model that can operate as either bi-encoder or cross-encoder
pub enum BertModel {
    BiEncoder(BertBiEncoder),
    CrossEncoder(BertCrossEncoder),
}

impl BertModel {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained(model_type: BertModelType) -> Result<Self> {
        use crate::bertconfig::BertConfig;
        use crate::bertweights::BertModelWeights;

        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("edgebert");

        std::fs::create_dir_all(&cache_dir)?;

        let (model_path, is_cross_encoder) = match model_type {
            BertModelType::MiniLML6V2BiEncoder => (cache_dir.join("all-MiniLM-L6-v2"), false),
            BertModelType::MiniLML6V2CrossEncoder => (cache_dir.join("ms-marco-MiniLM-L-6-v2"), true),
        };

        ensure_model_files(model_type, &model_path)?;

        let weights = BertModelWeights::load(&model_path)?;
        let config: BertConfig = serde_json::from_str(&weights.config_json)?;

        let tokenizer_file = model_path.join("tokenizer.json");
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        if let Some(padding) = tokenizer.get_padding_mut() {
            padding.strategy = tokenizers::PaddingStrategy::BatchLongest;
        }
        if let Some(truncation) = tokenizer.get_truncation_mut() {
            truncation.max_length = 512;
        }

        if is_cross_encoder {
            Ok(BertModel::CrossEncoder(BertCrossEncoder::from_weights(
                weights, tokenizer, config,
            )?))
        } else {
            Ok(BertModel::BiEncoder(BertBiEncoder::from_weights(
                weights, tokenizer, config,
            )?))
        }
    }

    pub fn encode(&self, texts: Vec<&str>, normalize: bool) -> Result<Vec<Vec<f32>>> {
        match self {
            BertModel::BiEncoder(model) => model.encode(texts, normalize),
            BertModel::CrossEncoder(_) => {
                anyhow::bail!("Cross-encoder does not support encode operation")
            }
        }
    }
    pub fn score_batch(&mut self, pairs: Vec<(&str, &str)>) -> Result<Vec<f32>> {
        match self {
            BertModel::CrossEncoder(model) => model.score_batch(pairs),
            BertModel::BiEncoder(_) => {
                anyhow::bail!("Bi-encoder does not support score_batch operation")
            }
        }
    }
    pub fn score_pair(&mut self, query: &str, document: &str) -> Result<f32> {
        match self {
            BertModel::CrossEncoder(model) => model.score_pair(query, document),
            BertModel::BiEncoder(_) => {
                anyhow::bail!("Bi-encoder does not support score_pair operation")
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_model_files(model_type: BertModelType, model_path: &std::path::Path) -> Result<()> {
    let (weights_url, tokenizer_url, config_url) = match model_type {
        BertModelType::MiniLML6V2BiEncoder => (
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
        ),
        BertModelType::MiniLML6V2CrossEncoder => (
            "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/model.safetensors",
            "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json", 
            "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/config.json",
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
