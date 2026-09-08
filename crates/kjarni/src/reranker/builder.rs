//! Builder pattern for Reranker configuration.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_transformers::WgpuContext;

use crate::common::{DownloadPolicy, KjarniDevice, LoadConfig, LoadConfigBuilder};

use super::model::Reranker;
use super::presets::RerankerPreset;
use super::types::{RerankOverrides, RerankerResult};

/// Builder for configuring and constructing a Reranker.
pub struct RerankerBuilder {
    // Model identification
    pub(crate) model: String,
    pub(crate) model_path: Option<PathBuf>,

    // Execution environment
    pub(crate) device: KjarniDevice,
    pub(crate) context: Option<Arc<WgpuContext>>,
    pub(crate) cache_dir: Option<PathBuf>,

    // Loading configuration
    pub(crate) load_config: Option<LoadConfig>,

    // Download policy
    pub(crate) download_policy: DownloadPolicy,

    // Reranking defaults
    pub(crate) overrides: RerankOverrides,

    // Behavior
    pub(crate) quiet: bool,
}

impl RerankerBuilder {
    /// Create a new builder for the specified model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_path: None,
            device: KjarniDevice::default(),
            context: None,
            cache_dir: None,
            load_config: None,
            download_policy: DownloadPolicy::default(),
            overrides: RerankOverrides::default(),
            quiet: false,
        }
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &RerankerPreset) -> Self {
        Self {
            model: preset.model.to_string(),
            model_path: None,
            device: preset.recommended_device,
            context: None,
            cache_dir: None,
            load_config: None,
            download_policy: DownloadPolicy::default(),
            overrides: RerankOverrides::default(),
            quiet: false,
        }
    }

    /// Run on CPU (default).
    pub fn cpu(mut self) -> Self {
        self.device = KjarniDevice::Cpu;
        self
    }

    /// Run on GPU.
    pub fn gpu(mut self) -> Self {
        self.device = KjarniDevice::Gpu;
        self
    }

    /// Auto-select device.
    pub fn auto_device(mut self) -> Self {
        self.device = KjarniDevice::Auto;
        self
    }

    /// Provide a pre-created WgpuContext.
    pub fn with_context(mut self, context: Arc<WgpuContext>) -> Self {
        self.context = Some(context);
        self.device = KjarniDevice::Gpu;
        self
    }

    /// Set custom cache directory.
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Load model from a local path.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set model loading configuration.
    pub fn load_config(mut self, config: LoadConfig) -> Self {
        self.load_config = Some(config);
        self
    }

    /// Configure loading with a builder.
    pub fn with_load_config<F>(mut self, f: F) -> Self
    where
        F: FnOnce(LoadConfigBuilder) -> LoadConfigBuilder,
    {
        self.load_config = Some(f(LoadConfigBuilder::new()).build());
        self
    }

    /// Set download policy.
    pub fn download_policy(mut self, policy: DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Never download models.
    pub fn offline(mut self) -> Self {
        self.download_policy = DownloadPolicy::Never;
        self
    }
    /// Set default top-k results to return.
    pub fn top_k(mut self, k: usize) -> Self {
        self.overrides.top_k = Some(k);
        self
    }

    /// Set minimum score threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.overrides.threshold = Some(threshold);
        self
    }

    /// Squash scores to 0..1 instead of returning raw cross-encoder logits.
    ///
    /// A deliberate divergence from torch, which returns the logit. Ranking is
    /// unaffected either way since the squash is monotonic; `threshold` is read
    /// on whichever scale is in use. For a single readable number without
    /// changing the reranker, use `RerankResult::probability()`.
    pub fn normalize_scores(mut self, normalize: bool) -> Self {
        self.overrides.normalize_scores = normalize;
        self
    }

    /// Apply full overrides.
    pub fn overrides(mut self, overrides: RerankOverrides) -> Self {
        self.overrides = overrides;
        self
    }

    /// Suppress non-error output.
    pub fn quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    /// Build the Reranker.
    pub async fn build(self) -> RerankerResult<Reranker> {
        Reranker::from_builder(self).await
    }
}

impl Reranker {
    /// Create a builder for a model.
    pub fn builder(model: impl Into<String>) -> RerankerBuilder {
        RerankerBuilder::new(model)
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &RerankerPreset) -> RerankerBuilder {
        RerankerBuilder::from_preset(preset)
    }
}
