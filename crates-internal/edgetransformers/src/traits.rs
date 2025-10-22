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
    type Config: ModelConfig;

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
    type Config: ModelConfig;

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
    type Config: ModelConfig;

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