//! The user-facing implementation for BERT models.
//!
//! This module defines the final `BertBiEncoder` struct that users will interact with.
//! It acts as a thin wrapper around the generic `TransformerEncoder` from `edgetransformers`,
//! configuring it specifically for the BERT architecture and providing the task-specific
//! `encode` method for generating sentence embeddings.

use anyhow::{Result, anyhow};
use edgetransformers::encoder::TransformerEncoder;
use edgetransformers::prelude::*;
use edgetransformers::weights::ModelWeights;
use ndarray::{Array2, Axis};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

use edgetransformers::wgpu_context::WgpuContext;
mod config;
pub use config::{BertConfig, RobertaConfig, BertBaseConfig};

/// An application wrapper that provides bi-encoder functionality.
///
/// This struct composes a generic `TransformerEncoder` with a tokenizer and
/// the necessary pooling logic to produce sentence embeddings.
pub struct SentenceEncoder {
    base_encoder: TransformerEncoder,
    tokenizer: Tokenizer,
}


// todo: use 
#[derive(Debug, Clone)]
pub enum Model {
    // Sentence Transformers
    MiniLML6V2,
    MPNetBaseV2,
    
    // Base BERT models
    BertBaseUncased,
    BertBaseCased,
    BertLargeUncased,
    
    // RoBERTa
    RobertaBase,
    RobertaLarge,
    
    /// Custom model from HuggingFace Hub
    Custom(String),
}

impl SentenceEncoder { // todo: change to sentenceEncoder
    /// Creates a new `BiEncoder`
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        
        
        // let config: BertConfig = serde_json::from_str(&weights.config_json)?;
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

          // Auto-detect architecture from tensor names
        let sample_keys: Vec<_> = weights.tensors.keys().take(5).collect();
        
        let has_bert_prefix = sample_keys.iter().any(|k| k.starts_with("bert."));
        let has_gamma_beta = sample_keys.iter().any(|k| k.contains("gamma") || k.contains("beta"));
        let has_token_type = weights.tensors.contains_key("embeddings.token_type_embeddings.weight") 
            || weights.tensors.contains_key("bert.embeddings.token_type_embeddings.weight");

        let base_encoder = if has_bert_prefix && has_gamma_beta {
            println!("Detected: BERT (TensorFlow-style with bert. prefix)");
            let config: BertBaseConfig = serde_json::from_str(&weights.config_json)?;
            TransformerEncoder::new(&weights, Arc::new(config), device, context)?
        } else if has_token_type {
            println!("Detected: BERT (HuggingFace-style)");
            let config: BertConfig = serde_json::from_str(&weights.config_json)?;
            TransformerEncoder::new(&weights, Arc::new(config), device, context)?
        } else {
            println!("Detected: RoBERTa/MPNet");
            let config: RobertaConfig = serde_json::from_str(&weights.config_json)?;
            TransformerEncoder::new(&weights, Arc::new(config), device, context)?
        };
        Ok(Self {
            base_encoder,
            tokenizer,
        })
    }

    /// Encodes a batch of texts into sentence embeddings.
    pub async fn encode(&self, texts: Vec<&str>, normalize: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Create input tensors
        let mut input_ids = Array2::<f32>::zeros((batch_size, max_len));
        let mut attention_mask = Array2::<f32>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            for (j, &id) in encoding.get_ids().iter().enumerate() {
                input_ids[[i, j]] = id as f32;
            }
            for (j, &mask) in encoding.get_attention_mask().iter().enumerate() {
                attention_mask[[i, j]] = mask as f32;
            }
        }

        let encoder_output = self
            .base_encoder
            .forward(&input_ids, &attention_mask)
            .await?;
        let embeddings = edgetransformers::pooling::mean_pool(
            &encoder_output.last_hidden_state,
            &attention_mask,
        )?;

        let final_embeddings = if normalize {
            let norms = embeddings
                .mapv(|x| x.powi(2))
                .sum_axis(Axis(1))
                .mapv(|x| x.sqrt().max(1e-12));
            let norms_expanded = norms.insert_axis(Axis(1));
            embeddings / &norms_expanded
        } else {
            embeddings
        };

        Ok(final_embeddings
            .outer_iter()
            .map(|row| row.to_vec())
            .collect())
    }
}
