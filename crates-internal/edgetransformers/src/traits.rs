//! Core model traits and data structures for transformer architectures.
//!
//! This module defines the fundamental, asynchronous traits that all models
//! in the library should implement. It establishes a clear contract for encoders,
//! decoders, and encoder-decoder models, providing a flexible abstraction that
//! is agnostic to the underlying computation backend (CPU or WGPU).
//!
//! The design principles are:
//! 1.  **Asynchronous by Default**: All `forward` methods are `async` to natively
//!     support non-blocking GPU operations via WGPU. The `async-trait` crate
//!     is used to enable `async` functions in traits.
//! 2.  **Stateless Models, Stateful Caches**: Models themselves are immutable (`&self`)
//!     during inference. All mutable state required for generation, such as Key-Value
//!     caches, is managed externally in a `Cache` object, which is passed mutably
//!     (`&mut dyn Cache`). This improves safety and allows for parallel inference.
//! 3.  **Backend Agnostic**: The `Device` enum allows a model to know its backend,
//!     but the trait signatures themselves are generic. Implementations will handle
//!     dispatching to the correct CPU or GPU code internally.
//! 4.  **Composition over Implementation**: These traits define architectural patterns.
//!     A concrete model like BART would be a struct that contains an `Encoder`
//!     implementation and a `CrossAttentionDecoder` implementation.
//!
//! # Crate Dependencies
//! Make sure to add `async-trait` to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! async-trait = "0.1"
//! ```

use anyhow::Result;
use async_trait::async_trait;
use ndarray::{Array2, Array3, Array4};
use std::any::Any;

/// Supported computation backends.
///
/// A model is typically initialized for a specific device and will use that
/// device for all its computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// Execute computations on the CPU.
    Cpu,
    /// Execute computations on the GPU via WGPU.
    Wgpu,
}

/// A base marker trait for all models in the library.
///
/// Provides a common interface for identifying the model's computation device.
/// It requires `Send + Sync` to ensure models can be safely used across threads.
pub trait Model: Send + Sync {
    /// Returns the computation device this model instance is configured to use.
    fn device(&self) -> Device;
}

/// A marker trait for model configuration structs (e.g., `BertConfig`, `GptConfig`).
///
/// This allows for generic model loading and initialization from configuration data.
pub trait ModelConfig: Send + Sync + Any {}

/// A type-erased, thread-safe container for mutable inference state.
///
/// This is essential for efficient autoregressive generation. Concrete implementations
/// might store attention Key-Value tensors, beam search hypotheses, or other
/// intermediate state that needs to be preserved across generation steps.
///
/// The `AsAny` and `AsAnyMut` methods are crucial for downcasting a `&mut dyn Cache`
/// back to its concrete type within a model's `forward` implementation.
pub trait Cache: Send + Sync {
    /// Returns a reference to the underlying cache as a type-erased `Any` object.
    fn as_any(&self) -> &dyn Any;
    /// Returns a mutable reference to the underlying cache as a type-erased `Any` object.
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Get the current sequence length (number of cached tokens)
    fn get_seq_length(&self) -> usize;
    /// Clear the cache
    fn clear(&mut self);
}

/// The standard output from an encoder model.
pub struct EncoderOutput<T = f32> {
    /// The final hidden states of the encoder.
    /// Shape: `(batch_size, sequence_length, hidden_size)`.
    pub last_hidden_state: Array3<T>,
}

/// The standard output from a decoder model.
pub struct DecoderOutput<T = f32> {
    /// The final hidden states of the decoder.
    /// Shape: `(batch_size, sequence_length, hidden_size)`.
    pub last_hidden_state: Array3<T>,
    /// The updated Key-Value cache after this forward pass.
    /// This can be fed back into the next generation step.
    pub past_key_values: Option<Vec<(Array4<T>, Array4<T>)>>,
}

/// Defines the asynchronous interface for an encoder model (e.g., BERT).
///
/// An encoder processes an entire input sequence at once, creating a
/// rich, contextualized representation.
#[async_trait]
pub trait Encoder: Model {
    type Input;
    type Output;

    /// Asynchronously performs a forward pass through the encoder.
    ///
    /// # Arguments
    /// * `input` - The input tensor (e.g., token embeddings).
    /// * `attention_mask` - A mask to prevent attention to padding tokens.
    ///
    /// # Returns
    /// An `EncoderOutput` containing the final hidden states.
    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
    ) -> Result<Self::Output>;
}

/// Defines the asynchronous interface for a standalone decoder model (e.g., GPT-2).
///
/// A decoder is typically used for autoregressive generation, where it predicts
/// one token at a time. It uses a causal attention mask to ensure that a given
/// position can only attend to previous positions.
#[async_trait]
pub trait Decoder: Model {
    type Input;
    type Output;

    /// Asynchronously performs a forward pass through the decoder.
    ///
    /// # Arguments
    /// * `input` - The input tensor for the current step(s).
    /// * `attention_mask` - The causal attention mask.
    /// * `cache` - An optional mutable reference to a `Cache` object to enable
    ///   efficient, incremental decoding by reusing past Key-Value states.
    ///
    /// # Returns
    /// A `DecoderOutput` containing the new hidden states and the updated cache.
    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output>;
}

/// Defines the asynchronous interface for a decoder that uses cross-attention (e.g., BART Decoder).
///
/// This type of decoder attends to two sources: its own previously generated tokens
/// (self-attention) and the output of an encoder (cross-attention).
#[async_trait]
pub trait CrossAttentionDecoder: Model {
    type Input;
    type Output;

    /// Asynchronously performs a forward pass through the cross-attention decoder.
    ///
    /// # Arguments
    /// * `input` - The input tensor for the current decoder step(s).
    /// * `decoder_attention_mask` - The causal mask for the decoder's self-attention.
    /// * `encoder_output` - The output from an `Encoder` model, to be used in cross-attention.
    /// * `encoder_attention_mask` - The padding mask corresponding to the encoder's output.
    /// * `cache` - An optional mutable reference to the decoder's self-attention KV cache.
    ///
    /// # Returns
    /// A `DecoderOutput` containing the new hidden states and the updated cache.
    async fn forward(
        &self,
        input: &Self::Input,
        decoder_attention_mask: &Array2<f32>,
        encoder_output: &EncoderOutput,
        encoder_attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output>;
}

/// A trait providing high-level configuration shared by all transformer models.
///
/// This provides the essential hyperparameters needed to construct the layers
/// of a transformer model, such as the hidden dimensions and the number of
/// layers to build.
pub trait TransformerConfig: Send + Sync {
    /// The size of the hidden states and embedding dimensions.
    fn hidden_size(&self) -> usize;
    /// The number of attention heads in each multi-head attention layer.
    fn num_attention_heads(&self) -> usize;
    /// The total number of transformer layers (or blocks) in the model stack.
    fn num_hidden_layers(&self) -> usize;
    /// The epsilon value to use in LayerNorm layers for numerical stability.
    fn layer_norm_eps(&self) -> f32;
    ///
    fn is_causal(&self) -> bool;
    ///
    fn is_prenorm(&self) -> bool;
}

/// Describes the specific architectural details of an Encoder-only model (e.g., BERT, RoBERTa).
///
/// This trait acts as a "blueprint" that a generic `TransformerEncoder` can use to
/// construct itself. It provides a mapping from abstract component concepts (e.g., "the query
/// projection of the first layer's attention") to the concrete tensor names found in a
/// `safetensors` weight file.
pub trait EncoderArchitecture: TransformerConfig {
    /// Returns the tensor names for the word, position, and token type embeddings.
    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>); // RoBERTa has no token_type_embeddings

    /// Returns the tensor names for the LayerNorm applied after the embedding layer.
    fn get_embedding_layer_norm_names(&self) -> (&str, &str);

    /// Returns the names of all weights and biases for the attention component of a specific encoder layer.
    fn get_attention_names(&self, layer_index: usize) -> LayerAttentionNames;

    /// Returns the names of all weights and biases for the feed-forward component of a specific encoder layer.
    fn get_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;

    /// If we should transpose the feedforward weighs
    /// The most common convention and vajority of models do this and tre-transpose the weights in FeedForward::new
    /// The older GPT2 architecture doesn't do this
    fn transpose_ffn_weights(&self) -> bool;
}

/// Describes the architectural specifics of a Decoder-only model (e.g., GPT-2, Llama).
///
/// This trait will enable the creation of a generic `TransformerDecoder` for
/// autoregressive language models by providing the necessary weight tensor names.
pub trait DecoderArchitecture: TransformerConfig {
    fn transpose_ffn_weights(&self) -> bool;

    /// Returns the tensor names for the word and position embeddings.
    fn get_embedding_weight_names(&self) -> (&str, &str);
    /// Returns the tensor names for the final LayerNorm before the LM head.
    fn get_final_layer_norm_names(&self) -> (&str, &str);
    /// Returns the name of the language modeling head weight tensor, which projects to the vocabulary.
    fn get_lm_head_name(&self) -> &str;
    /// Returns the names for the single, combined QKV projection in a decoder layer's attention block.
    fn get_attention_names(&self, layer_index: usize) -> LayerDecoderAttentionNames;
    /// Returns the names for the feed-forward block in a decoder layer.
    fn get_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;
}

/// A container for the concrete tensor names of an attention block in a transformer layer.
///
/// An instance of this struct is returned by `EncoderArchitecture::get_attention_names`,
/// providing the generic `TransformerEncoder` with all the keys it needs to load the
/// weights for a single attention component from a `ModelWeights` object.
pub struct LayerAttentionNames {
    /// Weight tensor for the Query projection.
    pub q_weight: String,
    /// Bias tensor for the Query projection.
    pub q_bias: String,
    /// Weight tensor for the Key projection.
    pub k_weight: String,
    /// Bias tensor for the Key projection.
    pub k_bias: String,
    /// Weight tensor for the Value projection.
    pub v_weight: String,
    /// Bias tensor for the Value projection.
    pub v_bias: String,
    /// Weight tensor for the output projection.
    pub output_weight: String,
    /// Bias tensor for the output projection.
    pub output_bias: String,
    /// Weight tensor for the LayerNorm following the attention block.
    pub norm_weight: String,
    /// Bias tensor for the LayerNorm following the attention block.
    pub norm_bias: String,
}

/// A container for the concrete tensor names of a feed-forward block in a transformer layer.
///
/// An instance of this struct is returned by `EncoderArchitecture::get_feed_forward_names`,
/// providing the generic `TransformerEncoder` with all the keys it needs to load the
/// weights for a single feed-forward component from a `ModelWeights` object.
pub struct LayerFeedForwardNames {
    /// Weight tensor for the intermediate (first) dense layer.
    pub intermediate_weight: String,
    /// Bias tensor for the intermediate (first) dense layer.
    pub intermediate_bias: String,
    /// Weight tensor for the output (second) dense layer.
    pub output_weight: String,
    /// Bias tensor for the output (second) dense layer.
    pub output_bias: String,
    /// Weight tensor for the LayerNorm following the feed-forward block.
    pub norm_weight: String,
    /// Bias tensor for the LayerNorm following the feed-forward block.
    pub norm_bias: String,
}



/// A container for the concrete tensor names of a decoder's causal self-attention block.
///
/// This is often different from an encoder's attention, sometimes using a single
/// combined projection matrix for Q, K, and V.
pub struct LayerDecoderAttentionNames {
    /// Weight for the combined Query, Key, and Value projection.
    pub qkv_weight: String,
    /// Bias for the combined Query, Key, and Value projection.
    pub qkv_bias: String,
    /// Weight for the output projection.
    pub output_weight: String,
    /// Bias for the output projection.
    pub output_bias: String,
    /// Weight for the LayerNorm preceding the attention block.
    pub norm_weight: String,
    /// Bias for the LayerNorm preceding the attention block.
    pub norm_bias: String,
}

/// Describes the architectural specifics of an Encoder-Decoder model (e.g., BART, T5).
///
/// This trait will enable the creation of a generic `TransformerEncoderDecoder` for
/// sequence-to-sequence tasks. It provides methods to get tensor names for all
/// components: the shared embeddings, the encoder stack, and the decoder stack
/// (including its self-attention and cross-attention blocks).
pub trait EncoderDecoderArchitecture: TransformerConfig {
    /// Returns the name of the shared word embedding weight tensor.
    fn get_shared_embedding_weight_name(&self) -> &str;

    /// Returns the names for the encoder's specific components.
    fn get_encoder_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
    fn get_encoder_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;

    /// Returns the names for the decoder's self-attention block.
    fn get_decoder_self_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
    /// Returns the names for the decoder's cross-attention block, which attends to encoder outputs.
    fn get_decoder_cross_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
    /// Returns the names for the decoder's feed-forward block.
    fn get_decoder_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;

    /// Returns the name of the optional final bias tensor applied to the logits.
    /// (e.g., `final_logits_bias` in BART). Returns `None` if not present.
    fn get_final_logits_bias_name(&self) -> Option<&str>;
}
