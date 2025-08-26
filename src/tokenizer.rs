use wasm_bindgen::prelude::*;
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::HashMap;

#[cfg(target_arch = "wasm32")]
pub use self::wasm::WordPieceTokenizer;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Encoding {
        pub ids: Vec<u32>,
        pub attention_mask: Vec<u32>,
    }

    impl Encoding {
        pub fn get_attention_mask(&self) -> &Vec<u32> {
            &self.attention_mask
        }

        pub fn get_ids(&self) -> &Vec<u32> {
            &self.ids
        }

        pub fn len(&self) -> usize {
            self.ids.len()
        }
    }

    /// A lightweight, WASM-compatible that implements the WordPiece algorithm.
    pub struct WordPieceTokenizer {
        vocab: HashMap<String, u32>,
        unk_token: String,
        unk_token_id: u32,
        cls_token_id: u32,
        sep_token_id: u32,
        pad_token_id: u32,
    }

    impl WordPieceTokenizer {
        /// Creates a new tokenizer
        pub fn from_json_str(content: &str) -> Result<Self> {
            let json: Value = serde_json::from_str(content)?;

            // Extract the vocabulary
            let vocab = json["model"]["vocab"]
                .as_object()
                .ok_or_else(|| anyhow!("Could not find vocab in tokenizer json"))?
                .iter()
                .map(|(token, id)| (token.clone(), id.as_u64().unwrap() as u32))
                .collect::<HashMap<String, u32>>();

            // Helper to get a token and its ID or fail
            let get_token_info = |token_key: &str, json_val: &Value| -> Result<(String, u32)> {
                let token_str = json_val["added_tokens"]
                    .as_array()
                    .and_then(|tokens| {
                        tokens.iter().find(|t| t["content"].as_str() == Some(token_key))
                    })
                    .and_then(|t| t["content"].as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| token_key.to_string());

                let token_id = *vocab
                    .get(&token_str)
                    .ok_or_else(|| anyhow!("Token {} not found in vocabulary", token_str))?;

                Ok((token_str, token_id))
            };

            let (_, cls_token_id) = get_token_info("[CLS]", &json)?;
            let (_, sep_token_id) = get_token_info("[SEP]", &json)?;
            let (_, pad_token_id) = get_token_info("[PAD]", &json)?;
            let (unk_token, unk_token_id) = get_token_info("[UNK]", &json)?;

            Ok(Self {
                vocab,
                unk_token,
                unk_token_id,
                cls_token_id,
                sep_token_id,
                pad_token_id,
            })
        }
        pub fn from_file(path: &str) -> Result<Self> {
            let content = std::fs::read_to_string(path)?;
            Self::from_json_str(&content)
        }
        /// Tokenizes a word using WordPiece
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
        fn pre_tokenize(text: &str) -> Vec<String> {
            let mut tokens = Vec::new();
            let mut current = String::new();

            for ch in text.chars() {
                if ch.is_whitespace() {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                } else if ch.is_ascii_punctuation() {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    tokens.push(ch.to_string());
                } else {
                    current.push(ch.to_lowercase().next().unwrap());
                }
            }
            if !current.is_empty() {
                tokens.push(current);
            }
            tokens
        }
        /// Encodes a string
        pub fn encode(&self, text: &str, max_len: usize) -> Result<Encoding> {
            let mut ids = vec![self.cls_token_id];

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

            // WordPiece on each pre-tokenized word.
            for word in spaced_text.split_whitespace() {
                ids.extend(self.tokenize_word(word));
            }

            ids.push(self.sep_token_id);

            // Truncate
            if ids.len() > max_len {
                ids.truncate(max_len);
                ids[max_len - 1] = self.sep_token_id;
            }

            // Padd
            let current_len = ids.len();
            let mut attention_mask = vec![1; current_len];

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

        /// Encodes a batch of texts
        pub fn encode_batch(&self, texts: Vec<&str>, max_len: usize) -> Result<Vec<Encoding>> {
            texts.iter().map(|t| self.encode(t, max_len)).collect()
        }
    }
}