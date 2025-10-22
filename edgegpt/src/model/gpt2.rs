//! GPT-2 model implementation

use anyhow::Result;
use crate::config::GPTConfig;
use crate::weights::ModelWeights;
use crate::model::base::GPTBase;
use crate::generation::{GenerationConfig, generate_text};

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use crate::tokenizer::wasm::BPETokenizer as Tokenizer;

/// GPT-2 model for text generation
pub struct GPT2 {
    pub base: GPTBase,
    pub tokenizer: Tokenizer,
}

impl GPT2 {
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
    
    pub fn complete(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        let config = GenerationConfig {
            max_new_tokens: max_tokens,
            temperature,
            ..Default::default()
        };
        
        self.generate(prompt, &config)
    }
    
    pub fn batch_generate(
        &self,
        prompts: Vec<&str>,
        config: &GenerationConfig,
    ) -> Result<Vec<String>> {
        prompts.into_iter()
            .map(|prompt| self.generate(prompt, config))
            .collect()
    }
}