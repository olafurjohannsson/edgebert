//! Base traits and types for language model inference.

use crate::Cache;
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::GpuTensor;
pub use crate::tensor::DType;
use crate::traits::InferenceModel;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, ArrayView2, ArrayView3};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

/// Configuration for Rotary Position Embedding (RoPE) scaling.
///
/// Different families fill in different halves of this. Llama 3 supplies a single
/// `factor` with high and low frequency cutoffs. Phi-3 supplies LongRoPE, which is a
/// per-dimension factor list instead of one number, and names its strategy under `type`
/// rather than `rope_type`. Everything is therefore optional with a default, so that a
/// checkpoint carrying one style parses without inventing values for the other.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RopeScalingConfig {
    /// Global scaling factor applied to all frequencies. Llama 3 style.
    #[serde(default)]
    pub factor: f32,
    /// Scaling factor for high-frequency components.
    #[serde(default)]
    pub high_freq_factor: f32,
    /// Scaling factor for low-frequency components.
    #[serde(default)]
    pub low_freq_factor: f32,
    /// Maximum sequence length from base model training.
    #[serde(default)]
    pub original_max_position_embeddings: usize,
    /// Scaling strategy. HuggingFace writes this as `rope_type` for Llama and `type`
    /// for Phi-3, so both spellings are accepted.
    #[serde(default, alias = "type")]
    pub rope_type: String,
    /// LongRoPE per-dimension factors, used beyond the trained context. One entry per
    /// pair of head dimensions, so `head_dim / 2` of them.
    #[serde(default)]
    pub long_factor: Option<Vec<f32>>,
    /// LongRoPE per-dimension factors, used within the trained context.
    #[serde(default)]
    pub short_factor: Option<Vec<f32>>,
}

pub enum PaddingSide {
    Left,
    Right,
}

/// Defines the autoregressive generation loop strategy for decoder models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoregressiveLoop {
    /// uses the prefill output directly
    Pipelined,

    ///  two-pass logic for compatibility with legacy implementations.
    Legacy,
}

/// Flexible input type for model inference supporting multiple format
#[derive(Debug)]
pub enum ModelInput<'a> {
    /// Token IDs stored in GPU memory.
    #[cfg(not(target_arch = "wasm32"))]
    TokensGpu(&'a GpuTensor),

    /// Token IDs stored in CPU memory.
    TokensCpu(ndarray::ArrayView2<'a, u32>),

    /// Pre-computed hidden states stored in GPU memory.
    #[cfg(not(target_arch = "wasm32"))]
    HiddenGpu(&'a GpuTensor),

    /// Pre-computed hidden states stored in CPU memory.
    HiddenCpu(ndarray::ArrayView3<'a, f32>),
}
impl<'a> ModelInput<'a> {
    /// Creates a `ModelInput` from a slice of token IDs.
    pub fn from_tokens(tokens: &'a [u32]) -> Self {
        let view = ArrayView2::from_shape((1, tokens.len()), tokens)
            .expect("Failed to create token view from slice");
        ModelInput::TokensCpu(view)
    }

    /// Creates a `ModelInput` from a 2D array of token IDs.
    pub fn from_array(array: ArrayView2<'a, u32>) -> Self {
        ModelInput::TokensCpu(array)
    }

    /// Creates a `ModelInput` from a GPU tensor of token IDs.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_gpu_tokens(tensor: &'a GpuTensor) -> Self {
        ModelInput::TokensGpu(tensor)
    }

    /// Creates a `ModelInput` from pre-computed CPU hidden states.
    pub fn from_hidden(hidden: ArrayView3<'a, f32>) -> Self {
        ModelInput::HiddenCpu(hidden)
    }

    /// Creates a `ModelInput` from pre-computed GPU hidden states.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_gpu_hidden(tensor: &'a GpuTensor) -> Self {
        ModelInput::HiddenGpu(tensor)
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            ModelInput::TokensGpu(t) => t.shape()[0],
            ModelInput::TokensCpu(a) => a.shape()[0],
            #[cfg(not(target_arch = "wasm32"))]
            ModelInput::HiddenGpu(t) => t.shape()[0],
            ModelInput::HiddenCpu(a) => a.shape()[0],
        }
    }

    /// Returns the sequence length.
    pub fn seq_len(&self) -> usize {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::TokensCpu(a) => a.shape()[1],
            #[cfg(not(target_arch = "wasm32"))]
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(a) => a.shape()[1],
        }
    }

    /// Returns true if this is a token input (vs hidden states).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_tokens(&self) -> bool {
        matches!(self, ModelInput::TokensGpu(_) | ModelInput::TokensCpu(_))
    }

    /// Returns true if this is a token input (vs hidden states).
    #[cfg(target_arch = "wasm32")]
    pub fn is_tokens(&self) -> bool {
        matches!(self, ModelInput::TokensCpu(_))
    }

    /// Returns true if this input is on the GPU.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_gpu(&self) -> bool {
        matches!(self, ModelInput::TokensGpu(_) | ModelInput::HiddenGpu(_))
    }

    /// Returns true if this input is on the GPU.
    #[cfg(target_arch = "wasm32")]
    pub fn is_gpu(&self) -> bool {
        false
    }
}
/// Configuration for model loading and device placement.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelLoadConfig {
    /// Keep embedding layer on CPU to save VRAM (500MB-2GB).
    pub offload_embeddings: bool,
    /// Keep language model head on CPU to save VRAM.
    pub offload_lm_head: bool,
    /// Force quantization to this data type (overrides file format).
    pub target_dtype: Option<DType>,
    /// Quantize the language model head to this data type.
    pub quantize_lm_head: Option<DType>,
    /// Quantize the embeddings to this data type.
    pub quantize_embeddings: Option<DType>,
    /// Pre-allocate KV cache for this batch size.
    pub max_batch_size: Option<usize>,
    /// Pre-allocate KV cache for this sequence length.
    pub max_sequence_length: Option<usize>,
    /// Use gguf
    pub use_gguf: bool,
    /// Suppress download progress output.
    ///
    /// Defaults to false, so a load that has to fetch weights says so. The three
    /// pipeline loaders used to hardcode this to quiet, which meant a first call
    /// on an uncached model downloaded hundreds of megabytes in total silence and
    /// looked to the caller like a hang.
    pub quiet: bool,
}

impl ModelLoadConfig {
    /// Creates a configuration for full GPU execution.
    pub fn full_gpu() -> Self {
        Self::default()
    }

    /// Creates a configuration with embeddings offloaded to CPU.
    pub fn set_offload_embeddings() -> Self {
        Self {
            offload_embeddings: true,
            ..Default::default()
        }
    }

    /// Creates a configuration for quantized model loading
    pub fn quantized(dtype: DType) -> Self {
        Self {
            target_dtype: Some(dtype),
            ..Default::default()
        }
    }

    /// Sets quantization for the language model head
    pub fn with_quantized_lm_head(mut self, dtype: DType) -> Self {
        self.quantize_lm_head = Some(dtype);
        self
    }

    /// Sets whether to offload embeddings to CPU.
    pub fn with_offload_embeddings(mut self, offload: bool) -> Self {
        self.offload_embeddings = offload;
        self
    }

    /// Sets the target quantization data type
    pub fn with_target_dtype(mut self, dtype: DType) -> Self {
        self.target_dtype = Some(dtype);
        self
    }

    /// Sets the maximum batch size for KV cache pre-allocation
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = Some(size);
        self
    }

    /// Sets the maximum sequence length for KV cache pre-allocation
    pub fn with_max_sequence_length(mut self, len: usize) -> Self {
        self.max_sequence_length = Some(len);
        self
    }
}

/// Core trait for all language models providing tokenization and metadata.
#[async_trait]
pub trait LanguageModel: InferenceModel {
    /// Returns the vocabulary size of the model.
    fn vocab_size(&self) -> usize;

    /// Returns the hidden dimension size of the model.
    fn hidden_size(&self) -> usize;

    /// Returns the number of transformer layers in the model.
    fn num_layers(&self) -> usize;

    /// Returns the number of attention heads per layer.
    fn num_heads(&self) -> usize;

    /// Returns the maximum context size (sequence length) supported by the model.
    fn context_size(&self) -> usize;

    /// Returns a reference to the tokenizer used by this model.
    fn tokenizer(&self) -> &Tokenizer;

    /// Returns the end-of-sequence token ID if defined.
    fn eos_token_id(&self) -> Option<u32>;

    /// Returns multiple end-of-sequence token IDs if the model supports them.
    ///
    /// Some models define multiple EOS tokens (e.g., Llama 3 with `<|eot_id|>`).
    /// Default implementation returns `None`.
    fn eos_token_ids(&self) -> Option<Vec<u32>> {
        None
    }

    /// Returns the beginning-of-sequence token ID if defined.
    fn bos_token_id(&self) -> Option<u32>;

    /// Returns the forced beginning-of-sequence token ID for seq2seq models.
    ///
    /// Used by encoder-decoder models to start generation with a specific token.
    fn forced_bos_token_id(&self) -> Option<u32>;

    /// Returns the forced end-of-sequence token ID for seq2seq models.
    fn forced_eos_token_id(&self) -> Option<u32>;

    /// Returns the padding token ID used for batched inference.
    fn pad_token_id(&self) -> Option<u32>;

    /// Creates a new KV cache for autoregressive generation.
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        num_beams: usize,
    ) -> Result<Box<dyn Cache>>;

    /// Returns the set of token IDs that should stop generation.
    fn stop_token_ids(&self) -> std::collections::HashSet<u32> {
        let mut set = std::collections::HashSet::new();
        if let Some(id) = self.eos_token_id() {
            set.insert(id);
        }
        // Llama 3 specific EOT ID
        if let Some(eot) = self.tokenizer().token_to_id("<|eot_id|>") {
            set.insert(eot);
        }
        set
    }

    /// Returns the maximum generation length.
    fn max_length(&self) -> usize {
        self.context_size()
    }

    /// Tokenizes a single text string into token IDs.
    fn tokenize(&self, text: &str) -> Result<Array2<u32>> {
        let encoding = self
            .tokenizer()
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let ids = encoding.get_ids().to_vec();
        let seq_len = ids.len();
        Ok(Array2::from_shape_vec((1, seq_len), ids)?)
    }

    /// Tokenizes a batch of texts with padding to the maximum length.
    fn tokenize_batch(&self, texts: &[&str], side: PaddingSide) -> Result<Array2<u32>> {
        if texts.is_empty() {
            return Err(anyhow!("Cannot tokenize empty batch"));
        }

        let mut encodings = Vec::new();
        let mut max_len = 0;

        for text in texts {
            let encoding = self
                .tokenizer()
                .encode(*text, true)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

            max_len = max_len.max(encoding.len());
            encodings.push(encoding);
        }

        let pad_id = self.pad_token_id().unwrap_or(0);
        let batch_size = texts.len();
        let mut batch = Array2::from_elem((batch_size, max_len), pad_id);

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let len = ids.len();

            match side {
                PaddingSide::Right => {
                    // Standard for Encoders: [Token, Token, Pad, Pad]
                    for (j, &token_id) in ids.iter().enumerate() {
                        batch[[i, j]] = token_id;
                    }
                }
                PaddingSide::Left => {
                    // Standard for Decoders: [Pad, Pad, Token, Token]
                    let start_col = max_len - len;
                    for (j, &token_id) in ids.iter().enumerate() {
                        batch[[i, start_col + j]] = token_id;
                    }
                }
            }
        }

        Ok(batch)
    }

    /// Decodes token IDs back to text.
    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer()
            .decode(token_ids, true)
            .map_err(|e| anyhow!("Decoding failed: {}", e))
    }

    /// Decodes a batch of token ID sequences.
    fn decode_batch(&self, token_ids: &[Vec<u32>]) -> Result<Vec<String>> {
        token_ids.iter().map(|ids| self.decode(ids)).collect()
    }
}

#[cfg(test)]
mod rope_scaling_config_tests {
    use super::RopeScalingConfig;

    #[test]
    fn parses_llama3_style_scaling() {
        let json = r#"{
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }"#;
        let cfg: RopeScalingConfig = serde_json::from_str(json).expect("llama3 scaling");
        assert_eq!(cfg.rope_type, "llama3");
        assert_eq!(cfg.factor, 32.0);
        assert_eq!(cfg.original_max_position_embeddings, 8192);
        assert!(cfg.long_factor.is_none());
    }

    #[test]
    fn parses_phi3_longrope_which_shares_none_of_llama3s_fields() {
        // Phi-3 writes `type` rather than `rope_type` and supplies no `factor` at all.
        // Before these fields were optional this failed with "missing field `factor`",
        // which is what stopped the model loading.
        let json = r#"{
            "long_factor": [1.08, 1.11, 1.14],
            "short_factor": [1.0, 1.0, 1.0],
            "type": "longrope"
        }"#;
        let cfg: RopeScalingConfig = serde_json::from_str(json).expect("longrope scaling");
        assert_eq!(cfg.rope_type, "longrope");
        assert_eq!(
            cfg.short_factor.as_deref(),
            Some([1.0, 1.0, 1.0].as_slice())
        );
        assert_eq!(cfg.long_factor.as_ref().map(Vec::len), Some(3));
        // The Llama fields default rather than being invented.
        assert_eq!(cfg.factor, 0.0);
        assert_eq!(cfg.original_max_position_embeddings, 0);
    }

    #[test]
    fn an_empty_object_parses_to_defaults() {
        let cfg: RopeScalingConfig = serde_json::from_str("{}").expect("empty scaling");
        assert_eq!(cfg, RopeScalingConfig::default());
    }
}

#[cfg(test)]
mod load_config_tests {
    use super::*;

    /// Silence has to be opt-in. All three pipeline loaders used to hardcode
    /// `quiet: true`, so a first call on an uncached model downloaded hundreds of
    /// megabytes with no output and read as a hang. A `..Default::default()`
    /// somewhere flipping this back would restore that in silence, which is
    /// exactly the failure it caused the first time.
    #[test]
    fn downloads_are_visible_unless_asked_otherwise() {
        assert!(
            !ModelLoadConfig::default().quiet,
            "a default load must report that it is downloading"
        );
    }

    /// The GPU presets are about placement and precision; none of them is a reason
    /// to go quiet.
    #[test]
    fn presets_do_not_silence_downloads() {
        assert!(!ModelLoadConfig::full_gpu().quiet);
    }
}
