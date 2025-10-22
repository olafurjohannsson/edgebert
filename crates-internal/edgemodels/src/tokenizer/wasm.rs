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

/// A lightweight, WASM-compatible WordPiece tokenizer
pub struct WordPieceTokenizer {
    vocab: HashMap<String, u32>,
    unk_token_id: u32,
    cls_token_id: u32,
    sep_token_id: u32,
    pad_token_id: u32,
}

impl WordPieceTokenizer {
    pub fn from_json_str(content: &str) -> Result<Self> {
        let json: Value = serde_json::from_str(content)?;
        
        // Extract vocabulary
        let vocab = json["model"]["vocab"]
            .as_object()
            .ok_or_else(|| anyhow!("Could not find vocab in tokenizer json"))?
            .iter()
            .map(|(token, id)| (token.clone(), id.as_u64().unwrap() as u32))
            .collect::<HashMap<String, u32>>();
        
        // Helper to get token ID
        let get_token_id = |token: &str| -> Result<u32> {
            vocab.get(token)
                .copied()
                .ok_or_else(|| anyhow!("Token {} not found in vocabulary", token))
        };
        
        let cls_token_id = get_token_id("[CLS]")?;
        let sep_token_id = get_token_id("[SEP]")?;
        let pad_token_id = get_token_id("[PAD]")?;
        let unk_token_id = get_token_id("[UNK]")?;
        
        Ok(Self {
            vocab,
            unk_token_id,
            cls_token_id,
            sep_token_id,
            pad_token_id,
        })
    }
    
    /// Tokenize a word using WordPiece algorithm
    fn tokenize_word(&self, word: &str) -> Vec<u32> {
        if let Some(id) = self.vocab.get(word) {
            return vec![*id];
        }
        
        let mut sub_tokens = Vec::new();
        let mut remaining = word;
        
        while !remaining.is_empty() {
            let mut found = false;
            
            for i in (1..=remaining.len()).rev() {
                let prefix = &remaining[0..i];
                let token_to_check = if remaining.len() != word.len() {
                    format!("##{}", prefix)
                } else {
                    prefix.to_string()
                };
                
                if let Some(id) = self.vocab.get(&token_to_check) {
                    sub_tokens.push(*id);
                    remaining = &remaining[i..];
                    found = true;
                    break;
                }
            }
            
            if !found {
                return vec![self.unk_token_id];
            }
        }
        
        sub_tokens
    }
    
    /// Encode a string
    pub fn encode(&self, text: &str, max_len: usize) -> Result<Encoding> {
        let mut ids = vec![self.cls_token_id];
        
        // Simple pre-tokenization
        let mut spaced_text = String::new();
        for char in text.to_lowercase().chars() {
            if char.is_ascii_punctuation() {
                spaced_text.push(' ');
                spaced_text.push(char);
                spaced_text.push(' ');
            } else {
                spaced_text.push(char);
            }
        }
        
        // WordPiece tokenization
        for word in spaced_text.split_whitespace() {
            ids.extend(self.tokenize_word(word));
        }
        
        ids.push(self.sep_token_id);
        
        // Truncate if needed
        if ids.len() > max_len {
            ids.truncate(max_len);
            ids[max_len - 1] = self.sep_token_id;
        }
        
        // Create attention mask
        let current_len = ids.len();
        let mut attention_mask = vec![1; current_len];
        
        // Pad if needed
        if current_len < max_len {
            let padding_needed = max_len - current_len;
            ids.extend(vec![self.pad_token_id; padding_needed]);
            attention_mask.extend(vec![0; padding_needed]);
        }
        
        Ok(Encoding {
            ids,
            attention_mask,
        })
    }
    
    /// Encode a batch of texts
    pub fn encode_batch(&self, texts: Vec<&str>, max_len: usize) -> Result<Vec<Encoding>> {
        texts.iter()
            .map(|t| self.encode(t, max_len))
            .collect()
    }
}