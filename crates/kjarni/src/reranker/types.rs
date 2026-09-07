//! Types for the reranker module.

use std::fmt;
use thiserror::Error;

/// Errors specific to reranking.
#[derive(Debug, Error)]
pub enum RerankerError {
    /// Model not found.
    #[error("{0}")]
    UnknownModel(String),

    /// Model cannot rerank.
    #[error("Model '{model}' cannot be used for reranking: {reason}")]
    IncompatibleModel { model: String, reason: String },

    /// Model path not found.
    #[error("Model path not found: {0}")]
    ModelPathNotFound(String),

    /// Model not downloaded.
    #[error("Model '{0}' not downloaded and download policy is set to Never")]
    ModelNotDownloaded(String),

    /// Download failed.
    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Load failed.
    #[error("Failed to load model '{model}': {source}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Reranking failed.
    #[error("Reranking failed: {0}")]
    RerankingFailed(#[from] anyhow::Error),

    /// GPU unavailable.
    #[error("GPU requested but WebGPU context could not be created")]
    GpuUnavailable,

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for reranker operations.
pub type RerankerResult<T> = Result<T, RerankerError>;

/// A single reranking result.
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Original index in the input documents.
    pub index: usize,

    /// Relevance, 0..1, higher is more relevant.
    ///
    /// A probability rather than the cross-encoder's raw logit, unless
    /// `return_raw_scores` was set. Comparable across queries only loosely: it is
    /// calibrated for ranking, not as a confidence.
    pub score: f32,

    /// The document text.
    pub document: String,
}

impl RerankResult {
    /// Create a new rerank result.
    pub fn new(index: usize, score: f32, document: impl Into<String>) -> Self {
        Self {
            index,
            score,
            document: document.into(),
        }
    }
}

impl fmt::Display for RerankResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {:.4}: {}",
            self.index,
            self.score,
            if self.document.len() > 50 {
                format!("{}...", &self.document[..50])
            } else {
                self.document.clone()
            }
        )
    }
}

/// Overrides for reranking behavior.
#[derive(Debug, Clone, Default)]
pub struct RerankOverrides {
    /// Return only top-k results.
    pub top_k: Option<usize>,

    /// Minimum score threshold.
    pub threshold: Option<f32>,

    /// Return the cross-encoder's raw logit instead of a 0..1 probability.
    ///
    /// Off by default, so `score` is a probability and `threshold` is read on the
    /// same scale. Turn it on only to recover the underlying logit; ranking is
    /// identical either way, because the squash is monotonic.
    pub return_raw_scores: bool,

    /// Batch size for processing pairs.
    pub batch_size: Option<usize>,
}

impl RerankOverrides {
    /// Create overrides for search reranking (top 10).
    pub fn for_search() -> Self {
        Self {
            top_k: Some(10),
            threshold: None,
            return_raw_scores: false,
            batch_size: None,
        }
    }

    /// Create overrides for filtering (threshold-based).
    pub fn for_filtering(threshold: f32) -> Self {
        Self {
            top_k: None,
            threshold: Some(threshold),
            return_raw_scores: false,
            batch_size: None,
        }
    }

    /// Create overrides for top-k selection.
    pub fn top_k(k: usize) -> Self {
        Self {
            top_k: Some(k),
            threshold: None,
            return_raw_scores: false,
            batch_size: None,
        }
    }
}
