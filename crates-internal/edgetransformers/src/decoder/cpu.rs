use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3, Array4, s};
use std::sync::Arc;

use crate::traits::{
    Cache, Decoder, DecoderArchitecture, DecoderOutput, Device, Model, TransformerConfig,
};
use crate::weights::ModelWeights;
use crate::{Embeddings, FeedForward, LayerNorm, MultiHeadAttention, TransformerLayer};
use crate::utils::{create_causal_mask, create_padding_mask};

/// The CPU backend implementation for the generic `TransformerDecoder`.
pub struct CpuTransformerDecoder {
    embeddings: Embeddings,
    final_layer_norm: LayerNorm, // Changed from embeddings_layer_norm
    layers: Vec<TransformerLayer>,
    config: Arc<dyn DecoderArchitecture + Send + Sync>,
}

impl CpuTransformerDecoder {
    pub fn new<C>(weights: &ModelWeights, config: Arc<C>) -> Result<Self>
    where
        C: DecoderArchitecture + Send + Sync + 'static,
    {
        // Load embedding weights (no token_type for decoder)
        let (word_w, pos_w) = config.get_embedding_weight_names();
        let embeddings = Embeddings::new(
            weights.get_array2(word_w)?,
            weights.get_array2(pos_w)?,
            None, // No token_type_embeddings for decoder
        );

        // Get final layer norm (not embedding layer norm!)
        let (norm_w, norm_b) = config.get_final_layer_norm_names();
        let final_layer_norm = LayerNorm::new(
            weights.get_array1(norm_w)?,
            weights.get_array1(norm_b)?,
            config.layer_norm_eps(),
        );

        // Build each transformer layer
        // Build each transformer layer
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i);
            let ffn_names = config.get_feed_forward_names(i);

            // For GPT-2: QKV is combined, stored as [hidden, 3*hidden] - NO transpose needed!
            let qkv_weight = weights.get_array2(&attn_names.qkv_weight)?; // Remove .t()
            let qkv_bias = weights.get_array1(&attn_names.qkv_bias)?;

            let hidden_size = config.hidden_size();

            // Split along axis 1 (columns) into Q, K, V
            let q_weight = qkv_weight.slice(s![.., 0..hidden_size]).to_owned();
            let k_weight = qkv_weight
                .slice(s![.., hidden_size..2 * hidden_size])
                .to_owned();
            let v_weight = qkv_weight
                .slice(s![.., 2 * hidden_size..3 * hidden_size])
                .to_owned();

            let q_bias = qkv_bias.slice(s![0..hidden_size]).to_owned();
            let k_bias = qkv_bias.slice(s![hidden_size..2 * hidden_size]).to_owned();
            let v_bias = qkv_bias
                .slice(s![2 * hidden_size..3 * hidden_size])
                .to_owned();

            let attention = MultiHeadAttention::new(
                config.hidden_size(),
                config.num_attention_heads(),
                q_weight, // No .t() - GPT-2 weights are already [in, out]
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                weights.get_array2(&attn_names.output_weight)?, // No .t()
                weights.get_array1(&attn_names.output_bias)?,
            );

            let feed_forward = FeedForward::new(
                weights.get_array2(&ffn_names.intermediate_weight)?, // No .t()
                weights.get_array1(&ffn_names.intermediate_bias)?,
                weights.get_array2(&ffn_names.output_weight)?, // No .t()
                weights.get_array1(&ffn_names.output_bias)?,
            );

            // GPT-2 style: layer norm BEFORE attention (pre-norm)
            let layer_norm1 = LayerNorm::new(
                weights.get_array1(&attn_names.norm_weight)?,
                weights.get_array1(&attn_names.norm_bias)?,
                config.layer_norm_eps(),
            );

            let layer_norm2 = LayerNorm::new(
                weights.get_array1(&ffn_names.norm_weight)?,
                weights.get_array1(&ffn_names.norm_bias)?,
                config.layer_norm_eps(),
            );

            layers.push(TransformerLayer {
                attention,
                feedforward: feed_forward,
                layer_norm1,
                layer_norm2,
            });
        }

        Ok(Self {
            embeddings,
            final_layer_norm,
            layers,
            config: config as Arc<dyn DecoderArchitecture + Send + Sync>,
        })
    }

    fn embed_with_offset(&self, input_ids: &Array2<f32>, position_offset: usize) -> Array3<f32> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.config.hidden_size();

        let mut hidden_states = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));

        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                let word_emb = self.embeddings.word_embeddings.row(token_id);
                hidden_states.slice_mut(s![i, j, ..]).assign(&word_emb);
            }
        }

        // Add position embeddings with offset
        for j in 0..seq_len {
            let pos_idx = position_offset + j;
            let pos_emb = self.embeddings.position_embeddings.row(pos_idx);
            for i in 0..batch_size {
                hidden_states
                    .slice_mut(s![i, j, ..])
                    .scaled_add(1.0, &pos_emb);
            }
        }

        hidden_states
    }
}

impl Model for CpuTransformerDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
}

#[async_trait]
impl Decoder for CpuTransformerDecoder {
    type Input = Array2<f32>;
    type Output = DecoderOutput;

    async fn forward(
        &self,
        input_ids: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        let position_offset = if let Some(ref cache) = cache {
            cache.get_seq_length()
        } else {
            0
        };

        // Embed inputs
        let mut hidden_states = self.embed_with_offset(input_ids, position_offset);

        // Create causal mask if needed
        let mask = if self.config.is_causal() {
            let seq_len = input_ids.shape()[1];
            let total_len = position_offset + seq_len;
            create_causal_mask(total_len)
        } else {
            attention_mask.clone()
        };

        // Pass through transformer layers with config
        for layer in self.layers.iter() {
            hidden_states = layer.forward(
                hidden_states,
                &mask,
                self.config.as_ref(), // ← Add this!
            )?;
        }
        hidden_states = self.final_layer_norm.forward_3d(&hidden_states);

        Ok(DecoderOutput {
            last_hidden_state: hidden_states,
            past_key_values: None,
        })

        // // Pass through transformer layers (NO embedding layer norm for GPT-2!)
        // // TODO: Integrate cache for KV reuse
        // for (layer_idx, layer) in self.layers.iter().enumerate() {
        //     hidden_states = layer.forward(hidden_states, attention_mask)?;
        // }

        // // Apply final layer norm
        // hidden_states = self.final_layer_norm.forward_3d(&hidden_states);

        // Ok(DecoderOutput {
        //     last_hidden_state: hidden_states,
        //     past_key_values: None, // TODO: Extract from cache
        // })
    }
}
