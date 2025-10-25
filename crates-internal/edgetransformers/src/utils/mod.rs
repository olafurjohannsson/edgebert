//! Utility modules

pub mod linear_algebra;
pub mod masks;

pub use masks::{create_causal_mask, create_padding_mask};