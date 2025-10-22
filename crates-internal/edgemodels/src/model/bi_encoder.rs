//! Bi-encoder implementation for sentence embeddings

use anyhow::Result;
use ndarray::{Array2, Axis};
use lru::LruCache;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::num::NonZeroUsize;

use edgetransformers::pooling::mean_pool;
use crate::bertconfig::BertConfig;
use crate::bertweights::BertModelWeights;
use crate::model::bertbase::BertBase;

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use crate::tokenizer::wasm::WordPieceTokenizer as Tokenizer;

/// BERT bi-encoder for generating sentence embeddings
pub struct BertBiEncoder {
    base: BertBase,
    tokenizer: Tokenizer,
    #[cfg(not(target_arch = "wasm32"))]
    cache: Lazy<Mutex<LruCache<String, Vec<f32>>>>,
}

impl BertBiEncoder {
    pub fn from_weights(
        weights: BertModelWeights,
        tokenizer: Tokenizer,
        config: BertConfig,
    ) -> Result<Self> {
        
        let base = BertBase::from_weights(&weights, config, "".to_string())?;
        
        Ok(Self {
            base,
            tokenizer,
            #[cfg(not(target_arch = "wasm32"))]
            cache: Lazy::new(|| {
                Mutex::new(LruCache::new(
                    NonZeroUsize::new(1000).unwrap(),
                ))
            }),
        })
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    pub fn encode(&self, texts: Vec<&str>, normalize: bool) -> Result<Vec<Vec<f32>>> {
        use rayon::prelude::*;
        
        if texts.is_empty() {
            return Ok(vec![]);
        }
        
        let mut results = vec![None; texts.len()];
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();
        
        // Check cache
        {
            let mut cache = self.cache.lock().unwrap();
            for (i, text) in texts.iter().enumerate() {
                let cache_key = format!("{}||{}", text, normalize);
                if let Some(cached_embedding) = cache.get(&cache_key) {
                    results[i] = Some(cached_embedding.clone());
                } else {
                    uncached_indices.push(i);
                    uncached_texts.push(*text);
                }
            }
        }
        
        if uncached_texts.is_empty() {
            return Ok(results.into_iter().map(|r| r.unwrap()).collect());
        }
        
        // Encode uncached texts
        let new_embeddings = self.encode_uncached(uncached_texts, normalize)?;
        
        // Store in cache and results
        {
            let mut cache = self.cache.lock().unwrap();
            for (idx, embedding) in uncached_indices.iter().zip(new_embeddings) {
                let text = texts[*idx];
                let cache_key = format!("{}||{}", text, normalize);
                cache.put(cache_key, embedding.clone());
                results[*idx] = Some(embedding);
            }
        }
        
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
    
    #[cfg(target_arch = "wasm32")]
    pub fn encode(&mut self, texts: Vec<&str>, normalize: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        self.encode_uncached(texts, normalize)
    }
    
    fn encode_uncached(&self, texts: Vec<&str>, normalize: bool) -> Result<Vec<Vec<f32>>> {
        // Tokenize
        #[cfg(not(target_arch = "wasm32"))]
        let encodings = self.tokenizer
            .encode_batch(texts, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        #[cfg(target_arch = "wasm32")]
        let encodings = self.tokenizer.encode_batch(texts, 512)?;
        
        let batch_size = encodings.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }
        
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap();
        
        // Prepare inputs
        let mut input_ids = Array2::<f32>::zeros((batch_size, max_len));
        let mut attention_mask = Array2::<f32>::zeros((batch_size, max_len));
        
        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            
            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                input_ids[[i, j]] = id as f32;
                attention_mask[[i, j]] = m as f32;
            }
        }
        
        // Forward pass
        let hidden_states = self.base.forward(&input_ids, &attention_mask, None)?;
        
        // Mean pooling
        let embeddings = mean_pool(&hidden_states, &attention_mask)?;
        
        // Normalize if requested
        let final_embeddings = if normalize {
            let norms = embeddings.mapv(|x| x.powi(2)).sum_axis(Axis(1));
            let norms = norms.mapv(|x| x.sqrt().max(1e-12));
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
    
    /// Batch encoding for efficiency
    #[cfg(not(target_arch = "wasm32"))]
    pub fn encode_batch(
        &self,
        batches: Vec<Vec<&str>>,
        normalize: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        use rayon::prelude::*;
        
        batches
            .into_par_iter()
            .map(|batch| self.encode(batch, normalize))
            .collect()
    }
}