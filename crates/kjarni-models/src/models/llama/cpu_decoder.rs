//! CPU implementation of the Llama decoder architecture.

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::WgpuContext;
use std::sync::Arc;

use anyhow::Result;
use ndarray::{Array2, Array3};

use kjarni_transformers::{
    activations::Activation,
    cache::CpuKVCache,
    decoder::prelude::*,
    normalization::RMSNorm,
    pipeline::CpuLayerFactory,
    rope::RoPE,
    tensor::DType,
    traits::{Cache, Device, InferenceModel, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

pub struct LlamaCpuDecoder {
    pub layers: Vec<CpuRoPEDecoderLayer>,
    pub final_norm: RMSNorm,
    pub metadata: ModelMetadata,
}

impl LlamaCpuDecoder {
    pub fn new(
        weights: &ModelWeights,
        metadata: ModelMetadata,
        layout: ModelLayout,
        rope: Arc<RoPE>,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("llama layout must have a decoder section");

        let final_norm = RMSNorm::new(
            weights.get_array1(decoder_layout.final_norm_weight.as_ref().unwrap())?,
            metadata.norm_eps,
        );

        let mut layers = Vec::with_capacity(metadata.num_layers);
        for i in 0..metadata.num_layers {
            layers.push(Self::build_layer(
                weights,
                &metadata,
                &layout,
                i,
                rope.clone(),
                target_dtype,
            )?);
        }

        Ok(Self {
            layers,
            final_norm,
            metadata,
        })
    }

    fn build_layer(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        i: usize,
        rope: Arc<RoPE>,
        target_dtype: Option<DType>,
    ) -> Result<CpuRoPEDecoderLayer> {
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("llama layout must have a decoder section");
        let layer_layout = &decoder_layout.layer;
        let factory = CpuLayerFactory::new(weights).with_target_dtype(target_dtype);

        let attention: DecoderAttention =
            factory.build_decoder_attention(meta, &decoder_layout.layer.self_attn, i)?;

        let feed_forward =
            factory.build_swiglu_ffn(&decoder_layout.layer.ffn, Activation::SilU, i)?;

        let attention_norm = factory.build_norm(
            &layer_layout.self_attn.norm_weight,
            &layer_layout.self_attn.norm_bias,
            meta.norm_eps,
            i,
        )?;

        let ffn_norm = factory.build_norm(
            &layer_layout.ffn.norm_weight,
            &layer_layout.ffn.norm_bias,
            meta.norm_eps,
            i,
        )?;

        Ok(CpuRoPEDecoderLayer {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            rope,
        })
    }
}

impl InferenceModel for LlamaCpuDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuDecoder for LlamaCpuDecoder {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    // These read from metadata rather than returning 0, which is what they did
    // until a probe asked the decoder its own geometry and got "0 heads, hidden 0"
    // back for a 24-layer model. Nothing in the engine consumed them, so the
    // stubs were invisible; anything introspecting the model needs them true.
    fn head_dim(&self) -> usize {
        self.metadata.head_dim
    }

    fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }

    fn num_attention_heads(&self) -> usize {
        self.metadata.num_attention_heads
    }

    fn num_kv_heads(&self) -> usize {
        self.metadata.num_kv_heads
    }

    fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        Ok(self.final_norm.forward_3d(hidden_states))
    }

    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        let mut hidden = hidden_states.clone();
        let seq_len = hidden.shape()[1];

        let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuKVCache>());

        for i in start_layer..end_layer {
            let layer: &CpuRoPEDecoderLayer = &self.layers[i];

            if let Some(ref mut c) = cpu_cache_opt {
                let (k_full_mut, v_full_mut) = c.get_context_view_mut(i, seq_len)?;

                hidden = layer.forward(
                    &hidden,
                    attention_mask,
                    position_offset,
                    k_full_mut,
                    v_full_mut,
                )?;
            } else {
                let kv_dim = layer.attention.num_kv_heads * layer.attention.head_dim;
                let (b, s, _) = hidden.dim();

                let mut temp_k = Array3::<f32>::zeros((b, s, kv_dim));
                let mut temp_v = Array3::<f32>::zeros((b, s, kv_dim));

                hidden = layer.forward(
                    &hidden,
                    attention_mask,
                    position_offset,
                    temp_k.view_mut(),
                    temp_v.view_mut(),
                )?;
            }
        }

        if let Some(c) = cpu_cache_opt {
            c.increment_len(seq_len);
        }

        Ok(hidden)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let mut output = self.forward_layers(
            hidden_states,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )?;
        output = self.final_norm.forward_3d(&output);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use ndarray::ArrayView1;

    use super::*;
    use crate::models::llama::LlamaModel;

    use kjarni_transformers::{
        ModelType,
        weights::{ModelWeights, clear_mmap_cache},
    };

    fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    #[test]
    fn test_gguf_safetensors_alignment_parity() -> Result<()> {
        let gguf_path =
            Path::new("/home/olafurj/.cache/kjarni/olafuraron_distilbart-cnn-12-6/model-q4_k.gguf");
        let st_path = Path::new("/home/olafurj/.cache/kjarni/olafuraron_distilbart-cnn-12-6");

        if !gguf_path.exists() || !st_path.exists() {
            return Ok(());
        }

        let gguf_weights = ModelWeights::new(gguf_path)?;
        let st_weights = ModelWeights::new(st_path)?;

        let tensors_to_check = [
            ("model.layers.0.self_attn.q_proj.weight", true),
            ("model.layers.0.self_attn.k_proj.weight", true),
            ("model.layers.0.self_attn.v_proj.weight", false),
            ("model.layers.0.self_attn.o_proj.weight", false),
            ("model.layers.0.mlp.gate_proj.weight", false),
        ];

        for (name, is_interleaved) in tensors_to_check {
            if !st_weights.contains(name) || !gguf_weights.contains(name) {
                continue;
            }

            let st_arr = st_weights.get_array2(name)?;
            let gguf_arr = gguf_weights.get_array2(name)?;

            assert_eq!(
                gguf_arr.shape(),
                st_arr.shape(),
                "shape mismatch for {}",
                name
            );

            let rows_to_check = 64.min(gguf_arr.nrows());
            let mut total_sim = 0.0;

            for i in 0..rows_to_check {
                let sim = cosine_similarity(gguf_arr.row(i), st_arr.row(i));
                total_sim += sim;

                if sim < 0.80 {
                    if is_interleaved {
                        panic!(
                            "interleaved layer '{}' failed parity check at row {}",
                            name, i
                        );
                    } else {
                        panic!("standard layer '{}' failed parity check", name);
                    }
                }
            }

            let avg_sim = total_sim / rows_to_check as f32;
            assert!(avg_sim > 0.90, "average similarity too low for {}", name);
        }

        Ok(())
    }

    #[test]
    fn test_gguf_and_safetensors_load_identical_configs() -> Result<()> {
        let gguf_1b_path = Path::new(
            "/home/olafurj/.cache/kjarni/llama-3.2-1b-instruct-q4_k_m/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        );
        let st_1b_path = Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B-Instruct");

        let gguf_3b_path = Path::new(
            "/home/olafurj/.cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        );
        let st_3b_path = Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-3B-Instruct");

        if gguf_1b_path.exists() && st_1b_path.exists() {
            let model_gguf = LlamaModel::from_pretrained(
                gguf_1b_path,
                Device::Cpu,
                None,
                None,
                Some(ModelType::Llama3_2_1B_Instruct),
            )?;

            let model_st = LlamaModel::from_pretrained(
                st_1b_path,
                Device::Cpu,
                None,
                None,
                Some(ModelType::Llama3_2_1B_Instruct),
            )?;

            let config_gguf = model_gguf.config();
            let config_st = model_st.config();

            assert_eq!(config_gguf.hidden_size, config_st.hidden_size);
            assert_eq!(config_gguf.num_hidden_layers, config_st.num_hidden_layers);
            assert_eq!(
                config_gguf.num_attention_heads,
                config_st.num_attention_heads
            );
            assert_eq!(
                config_gguf.num_key_value_heads,
                config_st.num_key_value_heads
            );
            assert_eq!(config_gguf.intermediate_size, config_st.intermediate_size);
            assert_eq!(config_gguf.vocab_size, config_st.vocab_size);
            assert_eq!(
                config_gguf.max_position_embeddings,
                config_st.max_position_embeddings
            );
            assert!((config_gguf.rms_norm_eps - config_st.rms_norm_eps).abs() < 1e-6);
            assert_eq!(config_gguf.hidden_act, config_st.hidden_act);
            assert!((config_gguf.rope_theta - config_st.rope_theta).abs() < 1e-6);
            assert_eq!(config_gguf.pad_token_id, config_st.pad_token_id);
            assert_eq!(
                config_gguf.tie_word_embeddings,
                config_st.tie_word_embeddings
            );
        }

        clear_mmap_cache();

        if gguf_3b_path.exists() && st_3b_path.exists() {
            let model_gguf = LlamaModel::from_pretrained(
                gguf_3b_path,
                Device::Cpu,
                None,
                None,
                Some(ModelType::Llama3_2_3B_Instruct),
            )?;

            let model_st = LlamaModel::from_pretrained(
                st_3b_path,
                Device::Cpu,
                None,
                None,
                Some(ModelType::Llama3_2_3B_Instruct),
            )?;

            let config_gguf = model_gguf.config();
            let config_st = model_st.config();

            assert_eq!(config_gguf.hidden_size, config_st.hidden_size);
            assert_eq!(config_gguf.num_hidden_layers, config_st.num_hidden_layers);
            assert_eq!(
                config_gguf.num_attention_heads,
                config_st.num_attention_heads
            );
            assert_eq!(
                config_gguf.num_key_value_heads,
                config_st.num_key_value_heads
            );
            assert_eq!(config_gguf.intermediate_size, config_st.intermediate_size);
            assert_eq!(config_gguf.vocab_size, config_st.vocab_size);
            assert_eq!(
                config_gguf.max_position_embeddings,
                config_st.max_position_embeddings
            );
            assert!((config_gguf.rms_norm_eps - config_st.rms_norm_eps).abs() < 1e-6);
            assert_eq!(config_gguf.hidden_act, config_st.hidden_act);
            assert!((config_gguf.rope_theta - config_st.rope_theta).abs() < 1e-6);
            assert_eq!(config_gguf.pad_token_id, config_st.pad_token_id);
            assert_eq!(
                config_gguf.tie_word_embeddings,
                config_st.tie_word_embeddings
            );
        }

        clear_mmap_cache();

        Ok(())
    }
}
