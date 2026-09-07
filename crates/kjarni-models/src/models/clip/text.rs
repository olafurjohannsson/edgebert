//! CLIP text tower: a caption to a vector in the same space as an image.
//!
//! Structurally the vision tower with two changes, and both matter.
//!
//! Attention is **causal**: each token sees only what came before it. The shared
//! encoder stack cannot express that, because its mask is `[batch, seq_k]` and is
//! broadcast across every query row, so it can hide padding but not the future.
//! This builds on `MultiHeadAttention`, which takes an `is_causal` flag directly,
//! rather than widening the encoder's mask and paying for it on every embedding
//! model that will never need it.
//!
//! Pooling reads the **end-of-text position**, not position zero. Causality is why:
//! the last token is the only one that has seen the whole caption.

use anyhow::{Context, Result};
use kjarni_transformers::activations::Activation;
use kjarni_transformers::cpu::attention::multi_head_attention::MultiHeadAttention;
use kjarni_transformers::cpu::feedforward::StdFeedForward;
use kjarni_transformers::cpu::normalization::LayerNorm;
use kjarni_transformers::weights::ModelWeights;
use ndarray::{Array1, Array2, Array3};

use super::config::{ClipConfig, ClipTextConfig};

/// Loads a `[out, in]` weight as `[in, out]`.
fn transposed(weights: &ModelWeights, name: &str) -> Result<Array2<f32>> {
    Ok(weights
        .get_array2(name)
        .with_context(|| format!("CLIP text {name}"))?
        .reversed_axes()
        .as_standard_layout()
        .to_owned())
}

/// One pre-norm block: norm, attend, residual, norm, feed-forward, residual.
struct TextLayer {
    norm1: LayerNorm,
    attention: MultiHeadAttention,
    norm2: LayerNorm,
    ffn: StdFeedForward,
}

pub struct ClipTextModel {
    config: ClipTextConfig,
    token_embedding: Array2<f32>,
    position_embedding: Array2<f32>,
    layers: Vec<TextLayer>,
    final_layer_norm: LayerNorm,
    text_projection: Array2<f32>,
}

impl ClipTextModel {
    pub fn from_weights(config: &ClipConfig, weights: &ModelWeights) -> Result<Self> {
        let text = config.text()?.clone();
        let activation: Activation = config.text_activation()?;
        let eps = text.layer_norm_eps;

        let token_embedding = weights
            .get_array2("text_model.embeddings.token_embedding.weight")
            .context("CLIP text token_embedding")?;
        let position_embedding = weights
            .get_array2("text_model.embeddings.position_embedding.weight")
            .context("CLIP text position_embedding")?;

        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for i in 0..text.num_hidden_layers {
            let p = format!("text_model.encoder.layers.{i}");
            layers.push(TextLayer {
                norm1: LayerNorm::new(
                    weights.get_array1(&format!("{p}.layer_norm1.weight"))?,
                    weights.get_array1(&format!("{p}.layer_norm1.bias"))?,
                    eps,
                ),
                // `MultiHeadAttention` multiplies with `matmul_3d_2d`, which wants
                // `[in, out]`, while safetensors stores `[out, in]`. Every one of
                // these is square, so the shape assert inside cannot catch the
                // mistake: it just computes the wrong product. Transposed here.
                attention: MultiHeadAttention::new(
                    text.hidden_size,
                    text.num_attention_heads,
                    transposed(weights, &format!("{p}.self_attn.q_proj.weight"))?,
                    weights.get_array1(&format!("{p}.self_attn.q_proj.bias"))?,
                    transposed(weights, &format!("{p}.self_attn.k_proj.weight"))?,
                    weights.get_array1(&format!("{p}.self_attn.k_proj.bias"))?,
                    transposed(weights, &format!("{p}.self_attn.v_proj.weight"))?,
                    weights.get_array1(&format!("{p}.self_attn.v_proj.bias"))?,
                    transposed(weights, &format!("{p}.self_attn.out_proj.weight"))?,
                    weights.get_array1(&format!("{p}.self_attn.out_proj.bias"))?,
                    // No grouped-query attention in CLIP.
                    None,
                ),
                norm2: LayerNorm::new(
                    weights.get_array1(&format!("{p}.layer_norm2.weight"))?,
                    weights.get_array1(&format!("{p}.layer_norm2.bias"))?,
                    eps,
                ),
                ffn: StdFeedForward::new(
                    weights.get_array2(&format!("{p}.mlp.fc1.weight"))?,
                    weights.get_array1(&format!("{p}.mlp.fc1.bias"))?,
                    weights.get_array2(&format!("{p}.mlp.fc2.weight"))?,
                    weights.get_array1(&format!("{p}.mlp.fc2.bias"))?,
                    activation,
                ),
            });
        }

        let final_layer_norm = LayerNorm::new(
            weights.get_array1("text_model.final_layer_norm.weight")?,
            weights.get_array1("text_model.final_layer_norm.bias")?,
            eps,
        );
        let text_projection = weights
            .get_array2("text_projection.weight")
            .context("CLIP text_projection")?;

        Ok(Self {
            config: text,
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
            text_projection,
        })
    }

    pub fn config(&self) -> &ClipTextConfig {
        &self.config
    }

    /// Token ids to an L2-normalized vector in the shared space.
    ///
    /// `eos_index` is the position to pool from, which the caller knows because it
    /// built the sequence. CLIP takes the end-of-text token: under a causal mask
    /// it is the only position that has attended to the whole caption.
    pub fn embed_ids(&self, ids: &[u32], eos_index: usize) -> Result<Array1<f32>> {
        let seq = ids.len();
        let hidden = self.config.hidden_size;
        anyhow::ensure!(seq > 0, "cannot embed an empty token sequence");
        anyhow::ensure!(
            seq <= self.config.max_position_embeddings,
            "{seq} tokens exceeds CLIP's {} position embeddings",
            self.config.max_position_embeddings
        );
        anyhow::ensure!(
            eos_index < seq,
            "eos index {eos_index} outside {seq} tokens"
        );

        let mut h = Array3::<f32>::zeros((1, seq, hidden));
        for (t, &id) in ids.iter().enumerate() {
            let row = self
                .token_embedding
                .row(id as usize % self.token_embedding.shape()[0]);
            for k in 0..hidden {
                h[[0, t, k]] = row[k] + self.position_embedding[[t, k]];
            }
        }

        for layer in &self.layers {
            // Pre-norm: the block reads a normalized copy and the residual carries
            // the unnormalized stream, which is why `h` is not overwritten here.
            let normed = layer.norm1.forward_3d(&h);
            // `forward_self_attn` projects q/k/v, attends causally and applies the
            // output projection. `attend` alone expects already-projected inputs
            // and only reshapes them into heads, which silently produces plausible
            // garbage when handed raw hidden states.
            let (attended, _, _) = layer.attention.forward_self_attn(&normed, None, None)?;
            h = h + attended;

            let normed = layer.norm2.forward_3d(&h);
            let ffn_out = layer.ffn.forward(&normed)?;
            h = h + ffn_out;
        }

        let h = self.final_layer_norm.forward_3d(&h);

        let d = self.text_projection.shape()[0];
        let mut projected = Array1::<f32>::zeros(d);
        for i in 0..d {
            let w = self.text_projection.row(i);
            let mut acc = 0.0f32;
            for j in 0..hidden {
                acc += w[j] * h[[0, eos_index, j]];
            }
            projected[i] = acc;
        }

        let norm = projected.dot(&projected).sqrt();
        if norm > 0.0 {
            projected /= norm;
        }
        Ok(projected)
    }
}
