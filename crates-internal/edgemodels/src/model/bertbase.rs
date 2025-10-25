//! Base BERT model implementation

use crate::bertconfig::BertConfig;
use crate::bertweights::BertModelWeights;
use anyhow::Result;
use std::sync::Arc;
use edgetransformers::{Embeddings, FeedForward, LayerNorm, MultiHeadAttention, TransformerLayer};
use ndarray::{Array2, Array3};

/// Base BERT model with transformer layers
pub struct BertBase {
    pub embeddings: Embeddings,
    pub embeddings_layer_norm: LayerNorm,
    pub layers: Vec<TransformerLayer>,
    pub config: BertConfig,
}

impl BertBase {
    pub fn from_weights(
        weights: &BertModelWeights,
        config: BertConfig,
        layer_prefix: String,
    ) -> Result<Self> {
        // Load embeddings
        let word_embeddings = weights.get_array2(&format!(
            "{}{}",
            layer_prefix, "embeddings.word_embeddings.weight"
        ))?;
        let position_embeddings = weights.get_array2(&format!(
            "{}{}",
            layer_prefix, "embeddings.position_embeddings.weight"
        ))?;
        let token_type_embeddings = weights.get_array2(&format!(
            "{}{}",
            layer_prefix, "embeddings.token_type_embeddings.weight"
        ))?;

        let embeddings =
            Embeddings::new(word_embeddings, position_embeddings, Some(token_type_embeddings));

        let embeddings_layer_norm = LayerNorm::new(
            weights.get_array1(&format!("{}{}", layer_prefix, "embeddings.LayerNorm.weight"))?,
            weights.get_array1(&format!("{}{}", layer_prefix, "embeddings.LayerNorm.bias"))?,
            config.layer_norm_eps,
        );

        // Load transformer layers
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("{}{}{}", layer_prefix, "encoder.layer.", i);
            // Load attention weights
            let attention = MultiHeadAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                weights
                    .get_array2(&format!("{}.attention.self.query.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.attention.self.query.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.attention.self.key.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.attention.self.key.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.attention.self.value.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.attention.self.value.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.attention.output.dense.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.attention.output.dense.bias", prefix))?,
            );

            // Load feedforward weights
            let feedforward = FeedForward::new(
                weights
                    .get_array2(&format!("{}.intermediate.dense.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.intermediate.dense.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.output.dense.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.output.dense.bias", prefix))?,
            );

            // Load layer norms
            let layer_norm1 = LayerNorm::new(
                weights.get_array1(&format!("{}.attention.output.LayerNorm.weight", prefix))?,
                weights.get_array1(&format!("{}.attention.output.LayerNorm.bias", prefix))?,
                config.layer_norm_eps,
            );

            let layer_norm2 = LayerNorm::new(
                weights.get_array1(&format!("{}.output.LayerNorm.weight", prefix))?,
                weights.get_array1(&format!("{}.output.LayerNorm.bias", prefix))?,
                config.layer_norm_eps,
            );

            layers.push(TransformerLayer {
                attention,
                feedforward,
                layer_norm1,
                layer_norm2,
            });
        }

        Ok(Self {
            embeddings,
            embeddings_layer_norm,
            layers,
            config,
        })
    }

    /// Forward pass through BERT
    pub fn forward(
        &self,
        input_ids: &Array2<f32>,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        // Embed inputs
        let mut hidden = self.embeddings.forward(input_ids, token_type_ids);

        // Apply embeddings layer norm
        hidden = self.embeddings_layer_norm.forward_3d(&hidden);
        
        // Pass through transformer layers
        for layer in &self.layers {
            // hidden = layer.forward(hidden, attention_mask)?;
        }

        Ok(hidden)
    }
}
