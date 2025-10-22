//! BERT model implementations

pub mod bertbase;
pub mod bi_encoder;
pub mod cross_encoder;
pub mod bert;

use anyhow::Result;
use std::path::PathBuf;

pub use bertbase::BertBase;
pub use bi_encoder::BertBiEncoder;
pub use cross_encoder::BertCrossEncoder;
