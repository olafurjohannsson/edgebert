use edgetransformers::traits::{
    DecoderArchitecture, TransformerConfig, LayerDecoderAttentionNames, LayerFeedForwardNames,
};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct GPT2Config {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub layer_norm_epsilon: f32,
}

impl TransformerConfig for GPT2Config {
    fn hidden_size(&self) -> usize { self.n_embd }
    fn num_attention_heads(&self) -> usize { self.n_head }
    fn num_hidden_layers(&self) -> usize { self.n_layer }
    fn layer_norm_eps(&self) -> f32 { self.layer_norm_epsilon }
    fn is_causal(&self) -> bool { true }   // Causal attention
    fn is_prenorm(&self) -> bool { true }  // Pre-norm architecture
}

impl DecoderArchitecture for GPT2Config {
    fn transpose_ffn_weights(&self) -> bool { false }
    
    fn get_embedding_weight_names(&self) -> (&str, &str) {
        ("transformer.wte.weight", "transformer.wpe.weight")  // Added transformer. prefix
    }
    
    fn get_final_layer_norm_names(&self) -> (&str, &str) {
        ("transformer.ln_f.weight", "transformer.ln_f.bias")  // Added transformer. prefix
    }
    
    fn get_lm_head_name(&self) -> &str {
        "lm_head.weight"  // This one doesn't have the prefix
    }
    
    fn get_attention_names(&self, i: usize) -> LayerDecoderAttentionNames {
        LayerDecoderAttentionNames {
            qkv_weight: format!("transformer.h.{}.attn.c_attn.weight", i),  // Added transformer.
            qkv_bias: format!("transformer.h.{}.attn.c_attn.bias", i),
            output_weight: format!("transformer.h.{}.attn.c_proj.weight", i),
            output_bias: format!("transformer.h.{}.attn.c_proj.bias", i),
            norm_weight: format!("transformer.h.{}.ln_1.weight", i),
            norm_bias: format!("transformer.h.{}.ln_1.bias", i),
        }
    }
    
    fn get_feed_forward_names(&self, i: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            intermediate_weight: format!("transformer.h.{}.mlp.c_fc.weight", i),  // Added transformer.
            intermediate_bias: format!("transformer.h.{}.mlp.c_fc.bias", i),
            output_weight: format!("transformer.h.{}.mlp.c_proj.weight", i),
            output_bias: format!("transformer.h.{}.mlp.c_proj.bias", i),
            norm_weight: format!("transformer.h.{}.ln_2.weight", i),
            norm_bias: format!("transformer.h.{}.ln_2.bias", i),
        }
    }
}