//! BART Encoder-Decoder model implementations

use crate::config::BartConfig;
use crate::ModelWeights;
use anyhow::Result;
use ndarray::{s, Array2, Array3, Array4, Axis};

use edgetransformers::{FeedForward, LayerNorm, MultiHeadAttention};
use edgetransformers::wgpu_context::WgpuContext;

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use crate::tokenizer::wasm::BPETokenizer as Tokenizer;

/// BART layer

// A standard Transformer layer for the encoder.
pub struct BartEncoderLayer {
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: LayerNorm,
    ffn: FeedForward,
    ffn_layer_norm: LayerNorm,
}
impl BartEncoderLayer {
    pub async fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        context: &WgpuContext,
    ) -> Result<Array3<f32>> {
        let residual = hidden_states;
        let (self_attn_output, _) = self.self_attn.forward_bart(
            hidden_states,
            None,
            Some(attention_mask),
            false,
            None,
        )?;

        let hidden_states_after_attn = residual + &self_attn_output;
        let hidden_states_after_attn_ln = self
            .self_attn_layer_norm
            .forward_3d(&hidden_states_after_attn);
        let residual = &hidden_states_after_attn_ln;
        let ffn_output = self.ffn.forward_gpu(residual, context).await?;
        let hidden_states_before_norm = residual + &ffn_output;
        let final_hidden_states = self.ffn_layer_norm.forward_3d(&hidden_states_before_norm);

        Ok(final_hidden_states)
    }
}

pub struct BartEncoder {
    layers: Vec<BartEncoderLayer>,
}

impl BartEncoder {
    pub async fn forward(
        &self,
        mut hidden_states: Array3<f32>,
        attention_mask: &Array2<f32>,
        context: &WgpuContext,
    ) -> Result<Array3<f32>> {
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask, context).await?;
        }
        Ok(hidden_states)
    }
}

pub struct BartDecoderLayer {
    // Two attention blocks for encoder-decoder
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: LayerNorm,
    cross_attn: MultiHeadAttention,
    cross_attn_layer_norm: LayerNorm,
    ffn: FeedForward,
    ffn_layer_norm: LayerNorm,
}
type LayerCache = (Array4<f32>, Array4<f32>);
// The full cache for the decoder is a Vec of these LayerCaches.
type FullCache = Vec<LayerCache>;

impl BartDecoderLayer {
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_causal_mask: Option<&Array2<f32>>,
        encoder_attention_mask: &Array2<f32>,
        layer_past: Option<&LayerCache>,
    ) -> Result<(Array3<f32>, LayerCache)> {
        let residual = hidden_states;

        let (self_attn_output, present_self_attn_kv) = self.self_attn.forward_bart(
            hidden_states,
            None,
            decoder_causal_mask,
            true,                
            layer_past,
        )?;

        let mut hidden_states = residual + &self_attn_output;
        hidden_states = self.self_attn_layer_norm.forward_3d(&hidden_states);

        let residual = &hidden_states;
        let (cross_attn_output, _) = self.cross_attn.forward_bart(
            &hidden_states,
            Some(encoder_hidden_states),
            Some(encoder_attention_mask),
            false, // not causal
            None,  // No cache for cross-attention
        )?;
        hidden_states = residual + &cross_attn_output;
        hidden_states = self.cross_attn_layer_norm.forward_3d(&hidden_states);
        let residual = &hidden_states;
        let ffn_output = self.ffn.forward(&hidden_states)?;
        hidden_states = residual + &ffn_output;
        hidden_states = self.ffn_layer_norm.forward_3d(&hidden_states);

        Ok((hidden_states, present_self_attn_kv))
    }
}

pub struct BartDecoder {
    layers: Vec<BartDecoderLayer>,
}

impl BartDecoder {
    pub fn forward(
        &self,
        mut hidden_states: Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_causal_mask: Option<&Array2<f32>>,
        encoder_attention_mask: &Array2<f32>,
        past_cache: Option<&FullCache>,
    ) -> Result<(Array3<f32>, FullCache)> {
        let mut present_cache = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_past = past_cache.map(|cache| &cache[i]);
            let (new_hidden_states, new_layer_cache) = layer.forward(
                &hidden_states,
                encoder_hidden_states,
                decoder_causal_mask,
                encoder_attention_mask,
                layer_past,
            )?;

            hidden_states = new_hidden_states;
            present_cache.push(new_layer_cache);
        }
        Ok((hidden_states, present_cache))
    }
}

// The main model
pub struct BartModel {
    pub shared_embeddings: Array2<f32>, //  word embedding matrix
    pub encoder_pos_embeddings: Array2<f32>,
    pub decoder_pos_embeddings: Array2<f32>,
    pub encoder_embed_layer_norm: LayerNorm,
    pub decoder_embed_layer_norm: LayerNorm,
    pub encoder: BartEncoder,
    pub decoder: BartDecoder,
    pub lm_head: Array2<f32>, // projection to vocabulary
    pub config: BartConfig,
    pub tokenizer: Tokenizer,
}

impl BartModel {
    pub fn from_weights(
        weights: &ModelWeights,
        config: BartConfig,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let shared_embeddings = weights.get_array2("model.shared.weight")?;
        let encoder_pos_embeddings = weights.get_array2("model.encoder.embed_positions.weight")?;
        let decoder_pos_embeddings = weights.get_array2("model.decoder.embed_positions.weight")?;

        let encoder_embed_layer_norm = LayerNorm::new(
            weights.get_array1("model.encoder.layernorm_embedding.weight")?,
            weights.get_array1("model.encoder.layernorm_embedding.bias")?,
            config.layer_norm_epsilon,
        );
        let decoder_embed_layer_norm = LayerNorm::new(
            weights.get_array1("model.decoder.layernorm_embedding.weight")?,
            weights.get_array1("model.decoder.layernorm_embedding.bias")?,
            config.layer_norm_epsilon,
        );

        let lm_head = shared_embeddings.clone();

        let mut encoder_layers = Vec::new();
        for i in 0..config.encoder_layers {
            let prefix = format!("model.encoder.layers.{}", i);
            let attn = MultiHeadAttention::new(
                config.d_model,
                config.encoder_attention_heads,
                weights
                    .get_array2(&format!("{}.self_attn.q_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.q_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.k_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.k_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.v_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.v_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.out_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.out_proj.bias", prefix))?,
            );
            let self_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.self_attn_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.self_attn_layer_norm.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            let fc1_weight = weights.get_array2(&format!("{}.fc1.weight", prefix))?;
            let fc2_weight = weights.get_array2(&format!("{}.fc2.weight", prefix))?;

            let ffn = FeedForward::new(
                fc1_weight.t().to_owned(),
                weights.get_array1(&format!("{}.fc1.bias", prefix))?,
                fc2_weight.t().to_owned(),
                weights.get_array1(&format!("{}.fc2.bias", prefix))?,
            );
            let ffn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.final_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.final_layer_norm.bias", prefix))?,
                config.layer_norm_epsilon,
            );

            encoder_layers.push(BartEncoderLayer {
                self_attn: attn,
                self_attn_layer_norm,
                ffn,
                ffn_layer_norm,
            });
        }
        let encoder = BartEncoder {
            layers: encoder_layers,
        };

        let mut decoder_layers = Vec::new();
        for i in 0..config.decoder_layers {
            let prefix = format!("model.decoder.layers.{}", i);

            // --- START OF MODIFICATION ---

            // Load the Q-projection weights for the self-attention block
            let self_attn_q_proj_weight =
                weights.get_array2(&format!("{}.self_attn.q_proj.weight", prefix))?;

            let self_attn = MultiHeadAttention::new(
                config.d_model,
                config.decoder_attention_heads,
                self_attn_q_proj_weight.t().to_owned(), // Use the tensor we just loaded
                weights.get_array1(&format!("{}.self_attn.q_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.k_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.k_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.v_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.v_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.out_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.out_proj.bias", prefix))?,
            );

            let self_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.self_attn_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.self_attn_layer_norm.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            let cross_attn = MultiHeadAttention::new(
                config.d_model,
                config.decoder_attention_heads,
                weights
                    .get_array2(&format!("{}.encoder_attn.q_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.encoder_attn.q_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.encoder_attn.k_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.encoder_attn.k_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.encoder_attn.v_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.encoder_attn.v_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.encoder_attn.out_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.encoder_attn.out_proj.bias", prefix))?,
            );
            let cross_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.encoder_attn_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.encoder_attn_layer_norm.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            let fc1_weight_dec = weights.get_array2(&format!("{}.fc1.weight", prefix))?;
            let fc2_weight_dec = weights.get_array2(&format!("{}.fc2.weight", prefix))?;

            let ffn = FeedForward::new(
                fc1_weight_dec.t().to_owned(),
                weights.get_array1(&format!("{}.fc1.bias", prefix))?,
                fc2_weight_dec.t().to_owned(),
                weights.get_array1(&format!("{}.fc2.bias", prefix))?,
            );
            let ffn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.final_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.final_layer_norm.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            decoder_layers.push(BartDecoderLayer {
                self_attn,
                self_attn_layer_norm,
                cross_attn,
                cross_attn_layer_norm,
                ffn,
                ffn_layer_norm,
            });
        }
        let decoder = BartDecoder {
            layers: decoder_layers,
        };

        Ok(Self {
            shared_embeddings,
            encoder_pos_embeddings,
            decoder_pos_embeddings,
            encoder_embed_layer_norm,
            decoder_embed_layer_norm,
            encoder,
            decoder,
            lm_head,
            config,
            tokenizer,
        })
    }

    pub fn embed(&self, input_ids: &Array2<f32>, is_decoder: bool, past_len: usize) -> Array3<f32> {
        let (batch_size, seq_len) = input_ids.dim();
        let use_scaling = self.config.scale_embedding; // || self.config.normalize_embedding;

        let embed_scale = if use_scaling {
            (self.config.d_model as f32).sqrt()
        } else {
            1.0
        };
        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, self.config.d_model));
        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                let mut embedding_row = self.shared_embeddings.row(token_id).to_owned();

                // Apply the scaling factor to the word embeddings
                embedding_row *= embed_scale;
                hidden.slice_mut(s![i, j, ..]).assign(&embedding_row);
            }
        }

        let (pos_embeddings, layer_norm) = if is_decoder {
            (&self.decoder_pos_embeddings, &self.decoder_embed_layer_norm)
        } else {
            (&self.encoder_pos_embeddings, &self.encoder_embed_layer_norm)
        };
 
        let start_idx = past_len + 2; // Add the BART offset (could be issues for other models, todo: config?)
        let end_idx = past_len + seq_len + 2;

        let pos_embeddings_slice = pos_embeddings
            .slice(s![start_idx..end_idx, ..]) 
            .insert_axis(Axis(0));
        hidden += &pos_embeddings_slice;

        layer_norm.forward_3d(&hidden)
    }
}
