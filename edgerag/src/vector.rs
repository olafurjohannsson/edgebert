use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorStore {
    pub embeddings: Vec<Vec<f32>>,
    pub dimension: usize,
}

impl VectorStore {
    pub fn new(embeddings: Vec<Vec<f32>>) -> Result<Self> {
        if embeddings.is_empty() {
            return Ok(Self {
                embeddings: vec![],
                dimension: 0,
            });
        }

        let dimension = embeddings[0].len();

        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != dimension {
                return Err(anyhow::anyhow!(
                    "Embedding {} has dimension {} but expected {}",
                    i, emb.len(), dimension
                ));
            }
        }

        Ok(Self { embeddings, dimension })
    }

    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..a.len() {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let denominator = (norm_a.sqrt() * norm_b.sqrt()).max(1e-9);
        dot_product / denominator
    }

    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Vec<(usize, f32)> {
        if self.embeddings.is_empty() || query_embedding.len() != self.dimension {
            return vec![];
        }

        let mut similarities: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(idx, emb)| (idx, Self::cosine_similarity(query_embedding, emb)))
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(limit);
        similarities
    }
}