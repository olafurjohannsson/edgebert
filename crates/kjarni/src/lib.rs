pub mod chat;
pub mod classifier;
pub mod common;
pub mod embedder;
pub mod generation;
pub mod indexer;
pub mod reranker;
pub mod searcher;
pub mod seq2seq;
pub mod summarizer;
pub mod transcriber;
pub mod translator;
mod utils;

pub use kjarni_transformers::PoolingStrategy;
// Re-export main API
pub use utils::*;

pub use kjarni_models::SequenceClassifier;
pub use kjarni_models::models::cross_encoder::CrossEncoder;
pub use kjarni_models::models::sentence_encoder::SentenceEncoder;

pub use crate::classifier::Classifier;
pub use crate::embedder::Embedder;
pub use crate::indexer::Indexer;
pub use crate::reranker::Reranker;
pub use crate::searcher::Searcher;
pub use crate::summarizer::Summarizer;
pub use crate::translator::Translator;

// Re-export core types
pub use kjarni_transformers::WgpuContext;
pub use kjarni_transformers::models::base::DType;
pub use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};
pub use kjarni_transformers::traits::Device;

// Weight loaders, so `kjarni inspect` can read a file's own description of itself
// without the CLI depending on kjarni-transformers directly.
#[cfg(not(target_arch = "wasm32"))]
pub use kjarni_transformers::cache::PrefixCache;
pub use kjarni_transformers::weights::{GgufLoader, SafeTensorsLoader, WeightLoader};
// Re-export generation
pub use kjarni_transformers::common::{
    BeamSearchParams, DecodingStrategy, GenerationConfig, SamplingParams, SpeculationParams,
    StreamedToken, TokenType,
};

// Re-export chat
pub use kjarni_transformers::chat::{
    llama3::{Llama2ChatTemplate, Llama3ChatTemplate},
    templates::{ChatTemplate, Conversation, Message, Role},
};

pub use kjarni_transformers::decoder::generator::DecoderGenerator;
pub use kjarni_transformers::decoder::traits::DecoderLanguageModel;

// Re-export seq2seq generation (encoder-decoders)
pub use kjarni_transformers::encoder_decoder::EncoderDecoderGenerator;

pub use kjarni_transformers::cpu::encoder::traits::EncoderLanguageModel;

pub use kjarni_rag::{
    DocumentLoader, IndexConfig, IndexReader, IndexWriter, LoaderConfig, Progress,
    ProgressCallback, ProgressStage, SearchIndex, SplitterConfig, TextSplitter,
};

pub use kjarni_rag::MetadataFilter;

pub use kjarni_search::{Bm25Index, Chunk, ChunkMetadata, SearchMode, SearchResult, VectorStore};

// Re-export commonly used types from dependencies
pub use kjarni_transformers::prelude::*;

pub mod models {
    pub use kjarni_models::models::bart::model::BartModel;
    pub use kjarni_models::models::gpt2::Gpt2Model;
    pub use kjarni_models::models::llama::{LlamaConfig, LlamaModel};
    pub use kjarni_models::models::phi::{PhiConfig, PhiModel};
    pub use kjarni_models::models::qwen::{QwenConfig, QwenModel};
}
pub mod kjarni_config;
pub mod kjarni_config_loader;
pub mod kjarni_config_resolve;
pub mod registry;

pub mod generator;

// FFI module (feature-gated and public)

#[cfg(any(feature = "python", feature = "c-bindings"))]
pub mod ffi;

// Prelude
pub mod prelude {
    pub use crate::utils::*;
    pub use kjarni_transformers::models::ModelType;
    pub use kjarni_transformers::prelude::*;
}

/// Send a chat message
///
/// # Example
///
/// ```ignore
/// let response = kjarni::chat("llama3.2-1b", "Hello!").await?;
/// ```
pub async fn chat_send(model: &str, message: &str) -> chat::ChatResult<String> {
    chat::send(model, message).await
}

/// Encode text to embedding (convenience wrapper).
///
/// # Example
///
/// ```ignore
/// let embedding = kjarni::embed("minilm-l6-v2", "Hello world").await?;
/// ```
pub async fn embed(model: &str, text: &str) -> anyhow::Result<Vec<f32>> {
    let encoder = crate::embedder::Embedder::new(model).await?;
    let embedding = encoder.embed(text).await?;
    Ok(embedding)
}

/// Classify text (convenience wrapper).
/// # Example
/// ```ignore
/// let result = kjarni::classify("distilbert-base-uncased-finetuned-sst-2-english", "I love programming!").await?;
/// ```
pub async fn classify(
    model: &str,
    text: &str,
) -> anyhow::Result<crate::classifier::ClassificationResult> {
    let classifier = crate::classifier::Classifier::new(model).await?;
    let result = classifier.classify(text).await?;
    Ok(result)
}

/// Get the kjarni version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get build information.
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        git_hash: option_env!("GIT_HASH"),
        build_date: option_env!("BUILD_DATE"),
    }
}

/// Build metadata.
#[derive(Debug, Clone)]
pub struct BuildInfo {
    pub version: &'static str,
    pub git_hash: Option<&'static str>,
    pub build_date: Option<&'static str>,
}
