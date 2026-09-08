pub mod chat;
pub mod classify;
pub mod display;
pub mod embed;
pub mod generate;
#[cfg(feature = "image-io")]
pub mod image;
pub mod index;
pub mod inspect;
pub mod model;
pub mod rerank;
pub mod search;
pub mod similarity;
pub mod summarize;
pub mod transcribe;
pub mod translate;

mod util;
