use anyhow::Result;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::traits::{Decoder, Device};
use edgetransformers::weights::ModelWeights;
use edgetransformers::wgpu_context::WgpuContext;
use edgetransformers::{Cache, CpuKVCache};
use ndarray::{Array1, Array2, Array3};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

mod config;
pub use config::GPT2Config;

pub struct GPT2Model {
    decoder: TransformerDecoder,
    tokenizer: Tokenizer,
    lm_head: Array2<f32>, // Add LM head (weight-tied with embeddings)
}

impl GPT2Model {
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let config: GPT2Config = serde_json::from_str(&weights.config_json)?;
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let decoder = TransformerDecoder::new(&weights, Arc::new(config), device, context)?;

        // Load LM head (typically weight-tied with word embeddings)
        // Use transpose of word embeddings for projection to vocabulary
        let lm_head = weights.get_array2("transformer.wte.weight")?.t().to_owned();

        Ok(Self {
            decoder,
            tokenizer,
            lm_head,
        })
    }

    pub async fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: Option<f32>,
        top_k: Option<usize>,
    ) -> Result<String> {
        let temp = temperature.unwrap_or(1.0);

        // Tokenize prompt
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Create cache (DistilGPT2 has 6 layers)
        let mut cache = CpuKVCache::new(6);

        // Generate tokens
        for _ in 0..max_new_tokens {
            // Create input (last token only when using cache)
            let input_len = if cache.get_seq_length() == 0 {
                token_ids.len()
            } else {
                1
            };

            let input_ids = Array2::from_shape_vec(
                (1, input_len),
                token_ids[token_ids.len() - input_len..]
                    .iter()
                    .map(|&id| id as f32)
                    .collect(),
            )?;

            // Create causal mask
            let seq_len = cache.get_seq_length() + input_len;
            let mask = Array2::ones((1, seq_len));

            // Forward pass
            let output = self
                .decoder
                .forward(&input_ids, &mask, Some(&mut cache))
                .await?;

            // Get last hidden state and project to vocabulary
            // Get last hidden state
            let last_hidden = output
                .last_hidden_state
                .slice(ndarray::s![0, -1, ..])
                .to_owned();

            // Manual matmul: output[j] = sum_i(hidden[i] * lm_head[i,j])
            let vocab_size = self.lm_head.shape()[1];
            let mut logits = Array1::<f32>::zeros(vocab_size);
            for j in 0..vocab_size {
                let mut sum = 0.0;
                for i in 0..last_hidden.len() {
                    sum += last_hidden[i] * self.lm_head[[i, j]];
                }
                logits[j] = sum / temp;
            }

            // Apply temperature
            let mut logits = logits.mapv(|x| x / temp);

            // Sample next token
            let next_token = if let Some(k) = top_k {
                self.sample_top_k(&logits, k)
            } else {
                // Greedy sampling
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap()
            };

            token_ids.push(next_token);

            // Check for EOS
            if next_token == self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(50256) {
                break;
            }
        }

        // Decode
        let text = self
            .tokenizer
            .decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        Ok(text)
    }

    fn sample_top_k(&self, logits: &ndarray::Array1<f32>, k: usize) -> u32 {
        use rand::distributions::{Distribution, WeightedIndex};

        // Get top k indices
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);

        // Convert to probabilities
        let max_logit = indexed[0].1;
        let exp_logits: Vec<f32> = indexed.iter().map(|(_, l)| (l - max_logit).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|e| e / sum).collect();

        // Sample from distribution
        let dist = WeightedIndex::new(&probs).unwrap();
        let mut rng = rand::thread_rng();
        indexed[dist.sample(&mut rng)].0 as u32
    }
}
