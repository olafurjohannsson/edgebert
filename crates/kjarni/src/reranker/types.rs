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

    /// Relevance, higher is more relevant.
    ///
    /// The cross-encoder's raw logit, which is what torch returns and so runs
    /// roughly -11 to +11 rather than 0 to 1. Use [`Self::probability`] for a
    /// readable number, or build the reranker with `normalize_scores` to get one
    /// everywhere.
    ///
    /// Meaningful as an ordering rather than as a confidence: it is calibrated
    /// for ranking, and is only loosely comparable across queries.
    pub score: f32,

    /// The document text.
    pub document: String,
}

impl RerankResult {
    /// [`Self::score`] squashed to 0..1, for display.
    ///
    /// The score itself stays a logit so it matches torch; this is the readable
    /// form. Ordering is unchanged, because the squash is monotonic, so ranking
    /// on either gives the same answer.
    ///
    /// Returns the score unchanged if the reranker already normalized, since
    /// squashing twice would be wrong.
    pub fn probability(&self) -> f32 {
        if (0.0..=1.0).contains(&self.score) {
            self.score
        } else {
            crate::reranker::model::sigmoid(self.score)
        }
    }

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

    /// Squash scores to 0..1 with a logistic instead of returning raw logits.
    ///
    /// Off by default, because a raw logit is what every reference returns: the
    /// ms-marco checkpoint declares `Identity` as its activation, and both
    /// `transformers` and `sentence-transformers` give the same unsquashed value.
    /// Turning this on is a deliberate divergence from torch, and `threshold` is
    /// then read on the 0..1 scale too.
    ///
    /// Ranking is identical either way, because the squash is monotonic; only the
    /// numbers move. To convert a single score without changing the reranker, use
    /// [`RerankResult::probability`].
    pub normalize_scores: bool,

    /// Batch size for processing pairs.
    pub batch_size: Option<usize>,
}

impl RerankOverrides {
    /// Create overrides for search reranking (top 10).
    pub fn for_search() -> Self {
        Self {
            top_k: Some(10),
            threshold: None,
            normalize_scores: false,
            batch_size: None,
        }
    }

    /// Create overrides for filtering (threshold-based).
    pub fn for_filtering(threshold: f32) -> Self {
        Self {
            top_k: None,
            threshold: Some(threshold),
            normalize_scores: false,
            batch_size: None,
        }
    }

    /// Create overrides for top-k selection.
    pub fn top_k(k: usize) -> Self {
        Self {
            top_k: Some(k),
            threshold: None,
            normalize_scores: false,
            batch_size: None,
        }
    }
}
