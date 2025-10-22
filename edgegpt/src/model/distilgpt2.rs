//! DistilGPT2 model implementation

use anyhow::Result;
use crate::config::GPTConfig;
use crate::weights::ModelWeights;
use crate::model::base::GPTBase;
use crate::generation::{GenerationConfig, generate_text};

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use crate::tokenizer::wasm::BPETokenizer as Tokenizer;

/// DistilGPT2 model for text generation
pub struct DistilGPT2 {
    pub base: GPTBase,
    pub tokenizer: Tokenizer,
}

impl DistilGPT2 {
    pub fn from_weights(
        weights: ModelWeights,
        tokenizer: Tokenizer,
        config: GPTConfig,
    ) -> Result<Self> {
        let base = GPTBase::from_weights(&weights, config)?;
        
        Ok(Self {
            base,
            tokenizer,
        })
    }
    
    pub fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        generate_text(
            &self.base,
            &self.tokenizer,
            prompt,
            config,
        )
    }
    
    pub fn get_embeddings(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize
        #[cfg(not(target_arch = "wasm32"))]
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        #[cfg(target_arch = "wasm32")]
        let encoding = self.tokenizer.encode(text, 512)?;
        
        let input_ids = encoding.get_ids();
        let batch_size = 1;
        let seq_len = input_ids.len();
        
        // Convert to ndarray
        let mut input_array = ndarray::Array2::<f32>::zeros((batch_size, seq_len));
        for (j, &id) in input_ids.iter().enumerate() {
            input_array[[0, j]] = id as f32;
        }
        
        // Forward pass
        let (hidden_states, _) = self.base.forward(&input_array, None)?;
        
        // Mean pooling over sequence dimension
        let embeddings = hidden_states.mean_axis(ndarray::Axis(1)).unwrap();
        
        Ok(embeddings.row(0).to_vec())
    }
}