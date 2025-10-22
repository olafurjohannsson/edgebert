//! WASM-compatible BPE tokenizer for GPT

use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Encoding {
    ids: Vec<u32>,
}

impl Encoding {
    pub fn get_ids(&self) -> &Vec<u32> {
        &self.ids
    }
    
    pub fn len(&self) -> usize {
        self.ids.len()
    }
}

/// BPE tokenizer for GPT models
pub struct BPETokenizer {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl BPETokenizer {
    pub fn from_json_str(content: &str) -> Result<Self> {
        let json: Value = serde_json::from_str(content)?;
        
        // Extract vocabulary
        let encoder = json["model"]["vocab"]
            .as_object()
            .ok_or_else(|| anyhow!("Could not find vocab in tokenizer json"))?
            .iter()
            .map(|(token, id)| (token.clone(), id.as_u64().unwrap() as u32))
            .collect::<HashMap<String, u32>>();
        
        let decoder: HashMap<u32, String> = encoder
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        
        // Extract merges
        let merges = json["model"]["merges"]
            .as_array()
            .ok_or_else(|| anyhow!("Could not find merges in tokenizer json"))?
            .iter()
            .filter_map(|merge| {
                merge.as_str().map(|s| {
                    let parts: Vec<&str> = s.split_whitespace().collect();
                    if parts.len() == 2 {
                        Some((parts[0].to_string(), parts[1].to_string()))
                    } else {
                        None
                    }
                })
            })
            .filter_map(|x| x)
            .collect();
        
        // Get special tokens
        let bos_token_id = encoder.get("<|startoftext|>").copied();
        let eos_token_id = encoder.get("<|endoftext|>").copied();
        let pad_token_id = encoder.get("<|pad|>").copied().or(eos_token_id);
        
        Ok(Self {
            encoder,
            decoder,
            merges,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }
    
    /// Simple BPE encoding
    pub fn encode(&self, text: &str, _max_len: usize) -> Result<Encoding> {
        let mut tokens = Vec::new();
        
        // Add BOS token if available
        if let Some(bos_id) = self.bos_token_id {
            tokens.push(bos_id);
        }
        
        // Basic byte-level tokenization
        let bytes = text.as_bytes();
        let mut word = Vec::new();
        
        for &byte in bytes {
            // Convert byte to special token representation
            let token = format!("Ġ{}", byte as char);
            if let Some(&id) = self.encoder.get(&token) {
                word.push(token);
            } else {
                // Fallback to individual byte
                let byte_token = format!("{}", byte as char);
                if let Some(&id) = self.encoder.get(&byte_token) {
                    word.push(byte_token);
                }
            }
            
            // Process at word boundaries (simplified)
            if byte == b' ' || byte == b'\n' {
                tokens.extend(self.bpe_merge(&word));
                word.clear();
            }
        }
        
        // Process remaining word
        if !word.is_empty() {
            tokens.extend(self.bpe_merge(&word));
        }
        
        Ok(Encoding { ids: tokens })
    }
    
    /// Apply BPE merges
    fn bpe_merge(&self, word: &[String]) -> Vec<u32> {
        if word.is_empty() {
            return vec![];
        }
        
        // For simplicity, just convert each token to ID
        word.iter()
            .filter_map(|token| self.encoder.get(token).copied())
            .collect()
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.decoder.get(&id).cloned())
            .collect();
        
        // Simple concatenation and cleanup
        let text = tokens.join("")
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .trim()
            .to_string();
        
        Ok(text)
    }
}