//! A "prelude" for the `edgetransformers` crate, re-exporting the most common traits and types.
//!
//! This allows users to easily import the essentials with a single `use` statement:
//! `use edgetransformers::prelude::*;`

pub use crate::traits::{
    // Core Traits
    Model,
    Encoder,
    Decoder,
    CrossAttentionDecoder,
    
    // Architectural Traits
    TransformerConfig,
    EncoderArchitecture,
    DecoderArchitecture,
    EncoderDecoderArchitecture,

    // Helper Structs for Traits
    LayerAttentionNames,
    LayerFeedForwardNames,
    LayerDecoderAttentionNames,
    
    // Data Structures
    EncoderOutput,
    DecoderOutput,
    
    // Backend Enum
    Device,
};