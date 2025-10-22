//! Base GPT model implementation

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, s, Axis};
use edgetransformers::{LayerNorm, MultiHeadAttention, FeedForward};
use crate::config::GPTConfig;
use crate::weights::ModelWeights;

/// GPT block containing attention and feedforward
pub struct GPTBlock {
    pub ln_1: LayerNorm,
    pub attn: CausalSelfAttention,
    pub ln_2: LayerNorm,
    pub mlp: FeedForward,
}

/// Causal self-attention for autoregressive modeling
pub struct CausalSelfAttention {
    pub c_attn_weight: Array2<f32>,  // Combined QKV projection
    pub c_attn_bias: Array1<f32>,
    pub c_proj_weight: Array2<f32>,  // Output projection
    pub c_proj_bias: Array1<f32>,
    pub n_head: usize,
    pub n_embd: usize,
    pub scale_factor: f32,
}

impl CausalSelfAttention {
    pub fn new(
        c_attn_weight: Array2<f32>,
        c_attn_bias: Array1<f32>,
        c_proj_weight: Array2<f32>,
        c_proj_bias: Array1<f32>,
        n_head: usize,
        n_embd: usize,
    ) -> Self {
        let head_dim = n_embd / n_head;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();
        
        Self {
            c_attn_weight,
            c_attn_bias,
            c_proj_weight,
            c_proj_bias,
            n_head,
            n_embd,
            scale_factor,
        }
    }
    
    pub fn forward(&self, x: &Array3<f32>, layer_past: Option<&(Array4<f32>, Array4<f32>)>) -> Result<(Array3<f32>, (Array4<f32>, Array4<f32>))> {
        let (batch_size, seq_len, _) = x.dim();
        let head_dim = self.n_embd / self.n_head;
        
        // Compute QKV with single matrix multiplication
        let qkv = edgetransformers::utils::linear_algebra::matmul_3d_2d(x, &self.c_attn_weight);

        let qkv = qkv + &self.c_attn_bias;
        
        // Split into Q, K, V
        let q = qkv.slice(s![.., .., ..self.n_embd]).to_owned();
        let k = qkv.slice(s![.., .., self.n_embd..2*self.n_embd]).to_owned();
        let v = qkv.slice(s![.., .., 2*self.n_embd..]).to_owned();
        
        // Reshape for multi-head attention
        let q = q.into_shape_with_order((batch_size, seq_len, self.n_head, head_dim))?
            .permuted_axes([0, 2, 1, 3]);
        let mut k = k.into_shape_with_order((batch_size, seq_len, self.n_head, head_dim))?
            .permuted_axes([0, 2, 1, 3]);
        let mut v = v.into_shape_with_order((batch_size, seq_len, self.n_head, head_dim))?
            .permuted_axes([0, 2, 1, 3]);
        
        // Handle past key-values for generation
        if let Some((past_k, past_v)) = layer_past {
            // Concatenate past and current
            let past_len = past_k.shape()[2];
            let total_len = past_len + seq_len;
            
            let mut k_combined = Array4::<f32>::zeros((batch_size, self.n_head, total_len, head_dim));
            let mut v_combined = Array4::<f32>::zeros((batch_size, self.n_head, total_len, head_dim));
            
            k_combined.slice_mut(s![.., .., ..past_len, ..]).assign(past_k);
            k_combined.slice_mut(s![.., .., past_len.., ..]).assign(&k);
            v_combined.slice_mut(s![.., .., ..past_len, ..]).assign(past_v);
            v_combined.slice_mut(s![.., .., past_len.., ..]).assign(&v);
            
            k = k_combined;
            v = v_combined;
        }
        
        // Store present key-values for next step
        let present_k = k.clone();
        let present_v = v.clone();
        
        // Compute attention scores
        let mut scores = edgetransformers::utils::linear_algebra::matmul_4d(&q, &k.permuted_axes([0, 1, 3, 2]));
        scores *= self.scale_factor;
        
        // Apply causal mask
        scores = apply_causal_mask(scores);
        
        // Softmax
        let weights = edgetransformers::activations::softmax(&scores);
        
        // Apply attention to values
        let context = edgetransformers::utils::linear_algebra::matmul_4d(&weights, &v);
        
        // Reshape back
        let context = context.permuted_axes([0, 2, 1, 3]);
        let context = context.as_standard_layout()
            .into_shape_with_order((batch_size, seq_len, self.n_embd))?
            .to_owned();
        
        // Output projection
        let mut output = edgetransformers::utils::linear_algebra::matmul_3d_2d(&context, &self.c_proj_weight);
        output += &self.c_proj_bias;
        
        Ok((output, (present_k, present_v)))
    }
}

/// Apply causal mask to attention scores (can only attend to previous positions)
fn apply_causal_mask(mut scores: Array4<f32>) -> Array4<f32> {
    let seq_len = scores.shape()[2];
    
    for i in 0..seq_len {
        for j in (i+1)..seq_len {
            scores.slice_mut(s![.., .., i, j]).fill(f32::NEG_INFINITY);
        }
    }
    
    scores
}

impl GPTBlock {
    pub fn forward(
        &self,
        x: Array3<f32>,
        layer_past: Option<&(Array4<f32>, Array4<f32>)>,
    ) -> Result<(Array3<f32>, (Array4<f32>, Array4<f32>))> {
        // Layer norm 1
        let ln1_out = self.ln_1.forward_3d(&x);
        
        // Self-attention with residual
        let (attn_out, present) = self.attn.forward(&ln1_out, layer_past)?;
        let x = x + attn_out;
        
        // Layer norm 2
        let ln2_out = self.ln_2.forward_3d(&x);
        
        // FFN with residual
        let mlp_out = self.mlp.forward(&ln2_out)?;
        let x = x + mlp_out;
        
        Ok((x, present))
    }
}

/// Base GPT model structure
pub struct GPTBase {
    pub wte: Array2<f32>,  // Word token embeddings
    pub wpe: Array2<f32>,  // Position embeddings
    pub blocks: Vec<GPTBlock>,
    pub ln_f: LayerNorm,   // Final layer norm
    pub config: GPTConfig,
}

impl GPTBase {
    pub fn from_weights(weights: &ModelWeights, config: GPTConfig) -> Result<Self> {
        // Load embeddings
        let wte = weights.get_array2("transformer.wte.weight")?;
        let wpe = weights.get_array2("transformer.wpe.weight")?;
        
        // Load transformer blocks
        let mut blocks = Vec::new();
        for i in 0..config.n_layer {
            let prefix = format!("h.{}", i);
            
            let ln_1 = LayerNorm::new(
                weights.get_array1(&format!("transformer.{}.ln_1.weight", prefix))?,
                weights.get_array1(&format!("transformer.{}.ln_1.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            
            let attn = CausalSelfAttention::new(
                weights.get_array2(&format!("transformer.{}.attn.c_attn.weight", prefix))?,
                weights.get_array1(&format!("transformer.{}.attn.c_attn.bias", prefix))?,
                weights.get_array2(&format!("transformer.{}.attn.c_proj.weight", prefix))?,
                weights.get_array1(&format!("transformer.{}.attn.c_proj.bias", prefix))?,
                config.n_head,
                config.n_embd,
            );
            
            let ln_2 = LayerNorm::new(
                weights.get_array1(&format!("transformer.{}.ln_2.weight", prefix))?,
                weights.get_array1(&format!("transformer.{}.ln_2.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            
            let mlp = FeedForward::new(
                weights.get_array2(&format!("transformer.{}.mlp.c_fc.weight", prefix))?,
                weights.get_array1(&format!("transformer.{}.mlp.c_fc.bias", prefix))?,
                weights.get_array2(&format!("transformer.{}.mlp.c_proj.weight", prefix))?,
                weights.get_array1(&format!("transformer.{}.mlp.c_proj.bias", prefix))?,
            );
            
            blocks.push(GPTBlock {
                ln_1,
                attn,
                ln_2,
                mlp,
            });
        }
        
        let ln_f = LayerNorm::new(
            weights.get_array1("transformer.ln_f.weight")?,
            weights.get_array1("transformer.ln_f.bias")?,
            config.layer_norm_epsilon,
        );
        
        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            config,
        })
    }
    
    pub fn forward(
        &self,
        input_ids: &Array2<f32>,
        past: Option<Vec<(Array4<f32>, Array4<f32>)>>,
    ) -> Result<(Array3<f32>, Vec<(Array4<f32>, Array4<f32>)>)> {
        let (batch_size, seq_len) = input_ids.dim();
        
        // Token embeddings
        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, self.config.n_embd));
        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                hidden.slice_mut(s![i, j, ..]).assign(&self.wte.row(token_id));
            }
        }
        
        // Position embeddings
        let past_len = past.as_ref().map(|p| p[0].0.shape()[2]).unwrap_or(0);
        for j in 0..seq_len {
            let pos = past_len + j;
            let mut slice = hidden.slice_mut(s![.., j, ..]);
            slice += &self.wpe.row(pos);
        }
        
        // Pass through transformer blocks
        let mut presents = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_past = past.as_ref().map(|p| &p[i]);
            let (new_hidden, present) = block.forward(hidden, layer_past)?;
            hidden = new_hidden;
            presents.push(present);
        }
        
        // Final layer norm
        hidden = self.ln_f.forward_3d(&hidden);
        
        Ok((hidden, presents))
    }
    
    pub fn get_logits(&self, hidden_states: &Array3<f32>) -> Array3<f32> {
        // Project to vocabulary
        edgetransformers::utils::linear_algebra::matmul_3d_2d(hidden_states, &self.wte.t().to_owned())
    }
}