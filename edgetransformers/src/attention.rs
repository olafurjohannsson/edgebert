//! Multi-head attention implementation

use crate::activations::softmax;
use crate::utils::linear_algebra::{
    apply_attention_mask, matmul_3d_2d, matmul_4d, matmul_4d_simple,
};
use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, Axis, Zip, s};

/// Multi-head attention mechanism
pub struct MultiHeadAttention {
    pub query_weight_t: Array2<f32>,
    pub query_bias: Array1<f32>,
    pub key_weight_t: Array2<f32>,
    pub key_bias: Array1<f32>,
    pub value_weight_t: Array2<f32>,
    pub value_bias: Array1<f32>,
    pub output_weight_t: Array2<f32>,
    pub output_bias: Array1<f32>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
}

impl MultiHeadAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        query_weight: Array2<f32>,
        query_bias: Array1<f32>,
        key_weight: Array2<f32>,
        key_bias: Array1<f32>,
        value_weight: Array2<f32>,
        value_bias: Array1<f32>,
        output_weight: Array2<f32>,
        output_bias: Array1<f32>,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();

        Self {
            query_weight_t: query_weight,
            query_bias,
            key_weight_t: key_weight,
            key_bias,
            value_weight_t: value_weight,
            value_bias,
            output_weight_t: output_weight,
            output_bias,
            num_heads,
            head_dim,
            scale_factor,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        // If this is Some, we perform cross-attention (encoder-decoder), else we perform self-attention (decoder)
        encoder_hidden_states: Option<&Array3<f32>>,
        attention_mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let batch_size = hidden_states.shape()[0];
        let seq_len = hidden_states.shape()[1];

        // Linear projections with bias

        // Project Q from decoders hidden states
        let mut q = matmul_3d_2d(hidden_states, &self.query_weight_t);
        q += &self.query_bias;

        // Determine source
        let kv_source = encoder_hidden_states.unwrap_or(hidden_states);
        let mut k = matmul_3d_2d(kv_source, &self.key_weight_t);
        k += &self.key_bias;
        let mut v = matmul_3d_2d(kv_source, &self.value_weight_t);
        v += &self.value_bias;

        // Reshape for multi-head attention
        let q = q
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let k_seq_len = kv_source.shape()[1];
        let k = k
            .into_shape_with_order((batch_size, k_seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let v = v
            .into_shape_with_order((batch_size, k_seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        // Compute attention scores
        let mut scores = matmul_4d(&q, &k.permuted_axes([0, 1, 3, 2]));
        scores *= self.scale_factor;

        // Apply mask

        if let Some(mask) = attention_mask {
            scores = apply_attention_mask(scores, mask);
        }

        // Softmax
        let weights = softmax(&scores);

        // Apply attention to values
        let context = matmul_4d(&weights, &v);

        // Reshape back
        let context = context.permuted_axes([0, 2, 1, 3]);
        let context = context
            .as_standard_layout()
            .into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        // Output projection
        let mut output = matmul_3d_2d(&context, &self.output_weight_t);
        output += &self.output_bias;

        Ok(output)
    }

    pub fn forward_bart(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: Option<&Array3<f32>>,
        attention_mask: Option<&Array2<f32>>,
        is_causal: bool,
        past_kv: Option<&(Array4<f32>, Array4<f32>)>,
        layer_idx: usize,
    ) -> Result<(Array3<f32>, (Array4<f32>, Array4<f32>))> {
        let batch_size = hidden_states.shape()[0];
        let query_len = hidden_states.shape()[1];

        let q_proj = matmul_3d_2d(hidden_states, &self.query_weight_t) + &self.query_bias;
        let kv_source = encoder_hidden_states.unwrap_or(hidden_states);
        let k_proj = matmul_3d_2d(kv_source, &self.key_weight_t) + &self.key_bias;
        let v_proj = matmul_3d_2d(kv_source, &self.value_weight_t) + &self.value_bias;
        let q_proj_scaled = &q_proj * self.scale_factor;


        let q = q_proj_scaled
            .into_shape_with_order((batch_size, query_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);
        let k_current_seq_len = kv_source.shape()[1];
        let mut k = k_proj
            .into_shape_with_order((batch_size, k_current_seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);
        let mut v = v_proj
            .into_shape_with_order((batch_size, k_current_seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        if let Some((past_k, past_v)) = past_kv {
            k = ndarray::concatenate(Axis(2), &[past_k.view(), k.view()])?;
            v = ndarray::concatenate(Axis(2), &[past_v.view(), v.view()])?;
        }
        let present_kv = (k.clone(), v.clone());
        let k_permuted = k.clone().permuted_axes([0, 1, 3, 2]);
        let q_contiguous = q.as_standard_layout().to_owned();
        let k_contiguous = k_permuted.as_standard_layout().to_owned();

        let mut scores = matmul_4d(&q_contiguous, &k_contiguous);

        if let Some(mask) = attention_mask {
            scores = if is_causal {
                let sliced_mask = mask.slice(s![..query_len, ..k.shape()[2]]);
                self.apply_causal_mask(scores, &sliced_mask.to_owned())
            } else {
                self.apply_padding_mask(scores, mask)
            };
        }
        let weights = softmax(&scores);
        let context = matmul_4d(&weights, &v.as_standard_layout().to_owned()); // Make v contiguous too, just to be safe
        let context = context.permuted_axes([0, 2, 1, 3]);
        let context = context
            .as_standard_layout()
            .into_shape_with_order((batch_size, query_len, self.num_heads * self.head_dim))?
            .to_owned();
        let output = matmul_3d_2d(&context, &self.output_weight_t) + &self.output_bias;

        Ok((output, present_kv))
    }
    pub fn apply_bart_attention_mask(
        &self,
        mut scores: Array4<f32>,
        mask: &Array2<f32>,
    ) -> Array4<f32> {
        // mask shape: [batch, seq_k]
        // scores shape: [batch, heads, seq_q, seq_k]
        // Need to broadcast mask to [batch, 1, 1, seq_k]

        let (batch_size, num_heads, seq_q, seq_k) = scores.dim();
        assert_eq!(mask.shape()[0], batch_size);
        assert_eq!(mask.shape()[1], seq_k);

        // Expand mask dimensions correctly for BART
        let mask_expanded = mask
            .view()
            .insert_axis(Axis(1)) // [batch, 1, seq_k]
            .insert_axis(Axis(1)); // [batch, 1, 1, seq_k]

        if let Some(broadcast_mask) = mask_expanded.broadcast((batch_size, num_heads, seq_q, seq_k))
        {
            Zip::from(&mut scores)
                .and(&broadcast_mask)
                .for_each(|s, &m| {
                    if m == 0.0 {
                        *s = f32::NEG_INFINITY;
                    }
                });
        }

        scores
    }
    /// Apply causal attention mask for decoder self-attention
    pub fn apply_causal_mask(&self, mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
        // mask shape: [seq_q, seq_k]
        // scores shape: [batch, heads, seq_q, seq_k]

        let (batch_size, num_heads, seq_q, seq_k) = scores.dim();
        assert_eq!(mask.shape()[0], seq_q);
        assert_eq!(mask.shape()[1], seq_k);

        // Expand: [seq_q, seq_k] -> [1, 1, seq_q, seq_k]
        let mask_expanded = mask
            .view()
            .insert_axis(Axis(0)) // [1, seq_q, seq_k]
            .insert_axis(Axis(0)); // [1, 1, seq_q, seq_k]

        if let Some(broadcast_mask) = mask_expanded.broadcast((batch_size, num_heads, seq_q, seq_k))
        {
            Zip::from(&mut scores)
                .and(&broadcast_mask)
                .for_each(|s, &m| {
                    if m == 0.0 {
                        *s = f32::NEG_INFINITY;
                    }
                });
        }

        scores
    }

    /// Apply padding attention mask (for encoder or cross-attention)
    pub fn apply_padding_mask(&self, mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
        // mask shape: [batch, seq_k]
        // scores shape: [batch, heads, seq_q, seq_k]

        let (batch_size, num_heads, seq_q, seq_k) = scores.dim();
        assert_eq!(mask.shape()[0], batch_size);
        assert_eq!(mask.shape()[1], seq_k);

        // Expand: [batch, seq_k] -> [batch, 1, 1, seq_k]
        let mask_expanded = mask
            .view()
            .insert_axis(Axis(1)) // [batch, 1, seq_k]
            .insert_axis(Axis(1)); // [batch, 1, 1, seq_k]

        if let Some(broadcast_mask) = mask_expanded.broadcast((batch_size, num_heads, seq_q, seq_k))
        {
            Zip::from(&mut scores)
                .and(&broadcast_mask)
                .for_each(|s, &m| {
                    if m == 0.0 {
                        *s = f32::NEG_INFINITY;
                    }
                });
        }

        scores
    }
}
