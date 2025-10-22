//! Cross-encoder implementation for pairwise scoring

use anyhow::Result;
use ndarray::{Array1, Array2};

use crate::config::BertConfig;
use crate::model::bertbase::BertBase;
use crate::weights::ModelWeights;
use edgetransformers::pooling::cls_pool;

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use crate::tokenizer::wasm::WordPieceTokenizer as Tokenizer;

/// BERT cross-encoder for scoring text pairs
pub struct BertCrossEncoder {
    base: BertBase,
    tokenizer: Tokenizer,
    classifier_weight: Array1<f32>,
    classifier_bias: f32,
}

// can be applied to raw logits to show a more user-friendly score
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl BertCrossEncoder {
    pub fn from_weights(
        weights: ModelWeights,
        tokenizer: Tokenizer,
        config: BertConfig,
    ) -> Result<Self> {
        let base = BertBase::from_weights(&weights, config, "bert.".to_string())?;

        // Load classifier head
        let classifier_weight_matrix = weights.get_array2("classifier.weight")?; // Shape: [1, 384]
        let classifier_weight = classifier_weight_matrix.row(0).to_owned(); // Shape: [384]
        let classifier_bias = weights.get_scalar("classifier.bias")?;
        Ok(Self {
            base,
            tokenizer,
            classifier_weight,
            classifier_bias,
        })
    }

    pub fn score_pair(&mut self, query: &str, document: &str) -> Result<f32> {
        let combined = format!("{} [SEP] {}", query, document);

        // Tokenize
        #[cfg(not(target_arch = "wasm32"))]
        let encodings = self
            .tokenizer
            .encode_batch(vec![combined.as_str()], true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        #[cfg(target_arch = "wasm32")]
        let encodings = self.tokenizer.encode_batch(vec![combined.as_str()], 512)?;

        let batch_size = 1;
        let seq_len = encodings[0].len();

        // Prepare inputs
        let mut input_ids = Array2::<f32>::zeros((batch_size, seq_len));
        let mut attention_mask = Array2::<f32>::zeros((batch_size, seq_len));

        for (j, (&id, &m)) in encodings[0]
            .get_ids()
            .iter()
            .zip(encodings[0].get_attention_mask().iter())
            .enumerate()
        {
            input_ids[[0, j]] = id as f32;
            attention_mask[[0, j]] = m as f32;
        }

        // Forward pass
        let hidden_states = self.base.forward(&input_ids, &attention_mask, None)?;

        // Extract CLS token
        let pooled = cls_pool(&hidden_states)?;
        let cls_output = pooled.row(0).to_owned();

        // Apply classifier head
        let logit: f32 = cls_output.dot(&self.classifier_weight) + self.classifier_bias;

        Ok(logit)
    }

    /// Batch scoring for efficiency
    pub fn score_batch(&mut self, pairs: Vec<(&str, &str)>) -> Result<Vec<f32>> {
        pairs.iter().map(|(q, d)| self.score_pair(q, d)).collect()
    }
}
