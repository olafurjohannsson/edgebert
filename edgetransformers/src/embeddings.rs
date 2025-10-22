//! Embedding layers for transformers

use ndarray::{Axis, Array2, Array3, s};

/// Configuration for embedding layers
pub struct EmbeddingConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
}

/// Combined embeddings (word + position + token type)
pub struct Embeddings {
    pub word_embeddings: Array2<f32>,
    pub position_embeddings: Array2<f32>,
    pub token_type_embeddings: Array2<f32>,
}

impl Embeddings {
    pub fn new(
        word_embeddings: Array2<f32>,
        position_embeddings: Array2<f32>,
        token_type_embeddings: Array2<f32>,
    ) -> Self {
        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
        }
    }
    
    /// Embed input tokens
    pub fn forward(
        &self,
        input_ids: &Array2<f32>,
        token_type_ids: Option<&Array2<f32>>,
    ) -> Array3<f32> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.word_embeddings.shape()[1];
        
        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));
        
        // Word embeddings
        #[cfg(not(target_arch = "wasm32"))]
        {
            use ndarray::parallel::prelude::*;
            hidden
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(input_ids.axis_iter(Axis(0)))
                .for_each(|(mut hidden_slice, ids)| {
                    for (j, &token_id) in ids.iter().enumerate() {
                        let word_emb = self.word_embeddings.row(token_id as usize);
                        hidden_slice.slice_mut(s![j, ..]).assign(&word_emb);
                    }
                });
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            for i in 0..batch_size {
                for j in 0..seq_len {
                    let token_id = input_ids[[i, j]] as usize;
                    let word_emb = self.word_embeddings.row(token_id);
                    hidden.slice_mut(s![i, j, ..]).assign(&word_emb);
                }
            }
        }
        
        // Position embeddings
        let pos_embeddings = self.position_embeddings.slice(s![0..seq_len, ..]);
        hidden += &pos_embeddings;
        
        // Token type embeddings (default to type 0)
        if let Some(type_ids) = token_type_ids {
            for i in 0..batch_size {
                for j in 0..seq_len {
                    let type_id = type_ids[[i, j]] as usize;
                    let type_emb = self.token_type_embeddings.row(type_id);
                    let mut slice = hidden.slice_mut(s![i, j, ..]);
                    slice += &type_emb;
                }
            }
        } else {
            let type_embeddings = self.token_type_embeddings.row(0);
            hidden += &type_embeddings;
        }
        
        hidden
    }
}