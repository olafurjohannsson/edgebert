//! Builder pattern for Generator configuration.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_transformers::WgpuContext;

use crate::common::{DownloadPolicy, KjarniDevice, LoadConfig, LoadConfigBuilder};
use crate::generation::GenerationOverrides;
use crate::generator::presets::GeneratorPreset;

use super::model::Generator;
use super::types::GeneratorResult;

/// Tokens a prefix cache holds unless the caller picks a size.
///
/// Sized against memory rather than the model's context window: KV is eagerly
/// allocated f32, roughly 24KB per token on Qwen2.5-0.5B, so this is about 96MB
/// there while that model's full 32768-token context would be 768MB. Long enough
/// to cover a real conversation or a RAG passage set.
pub const DEFAULT_PREFIX_CACHE_TOKENS: usize = 4096;

/// Builder for configuring a Generator.
///
/// # Example
///
/// ```ignore
/// let generator = Generator::builder("gpt2")
///     .cpu()
///     .temperature(0.8)
///     .max_tokens(100)
///     .build()
///     .await?;
/// ```
pub struct GeneratorBuilder {
    // Model selection
    pub(crate) model: String,
    pub(crate) model_path: Option<PathBuf>,

    // Device configuration
    pub(crate) device: KjarniDevice,
    pub(crate) context: Option<Arc<WgpuContext>>,

    // Model loading
    pub(crate) cache_dir: Option<PathBuf>,
    pub(crate) download_policy: DownloadPolicy,
    pub(crate) load_config: Option<LoadConfig>,

    // Speculative decoding: a small model proposes tokens, this one verifies a
    // batch of them in a single pass. Decode is bandwidth bound, so reading the
    // target's weights once for k tokens instead of k times is the saving.
    pub(crate) draft_model: Option<String>,
    pub(crate) draft_tokens: usize,

    // Prefix reuse: keep one KV cache alive across calls so a prompt that shares
    // a head with the last one prefills only the new tail. `None` is off.
    pub(crate) prefix_cache_tokens: Option<usize>,

    // Generation defaults
    pub(crate) generation_overrides: GenerationOverrides,

    // Behavior
    pub(crate) quiet: bool,
    pub(crate) allow_warnings: bool,
}

impl GeneratorBuilder {
    /// Create a new builder for the specified model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_path: None,
            device: KjarniDevice::default(),
            context: None,
            cache_dir: None,
            download_policy: DownloadPolicy::default(),
            load_config: None,
            draft_model: None,
            draft_tokens: 4,
            prefix_cache_tokens: None,
            generation_overrides: GenerationOverrides::default(),
            quiet: false,
            allow_warnings: false,
        }
    }

    /// Load model from a local path instead of registry.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the device for inference.
    pub fn device(mut self, device: KjarniDevice) -> Self {
        self.device = device;
        self
    }

    /// Use CPU for inference.
    pub fn cpu(self) -> Self {
        self.device(KjarniDevice::Cpu)
    }

    /// Use GPU for inference.
    pub fn gpu(self) -> Self {
        self.device(KjarniDevice::Gpu)
    }

    /// Provide a pre-created GPU context.
    pub fn with_context(mut self, context: Arc<WgpuContext>) -> Self {
        self.context = Some(context);
        self.device = KjarniDevice::Gpu;
        self
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &GeneratorPreset) -> Self {
        let mut builder = Self::new(preset.model);
        builder.device = preset.recommended_device;

        if let Some(temp) = preset.temperature {
            builder.generation_overrides.temperature = Some(temp);
        }
        builder.generation_overrides.max_new_tokens = Some(preset.default_max_tokens);

        builder
    }

    /// Configure for creative generation (high temperature).
    pub fn creative(mut self) -> Self {
        self.generation_overrides.temperature = Some(0.9);
        self.generation_overrides.top_p = Some(0.95);
        self
    }

    /// Configure for precise generation (low temperature).
    pub fn precise(mut self) -> Self {
        self.generation_overrides.temperature = Some(0.2);
        self.generation_overrides.top_p = Some(0.9);
        self
    }

    /// Enables speculative decoding with `name` as the draft model.
    ///
    /// The draft must share this model's vocabulary, which in practice means a
    /// smaller model of the same family: qwen2.5-0.5b drafting for qwen2.5-1.5b.
    /// `num_tokens` is how many the draft proposes per round; more trades wasted
    /// draft work against fewer passes over the target's weights.
    pub fn draft(mut self, name: impl Into<String>, num_tokens: usize) -> Self {
        self.draft_model = Some(name.into());
        self.draft_tokens = num_tokens.max(1);
        self
    }

    /// Reuses the KV cache across calls, prefilling only what a new prompt does
    /// not already share with the last one.
    ///
    /// Pays off whenever successive prompts share a long head: a chat replaying
    /// its history, or RAG re-sending the same retrieved passages. On
    /// Qwen2.5-0.5B a 2048 token prompt costs 15.96s cold and 0.49s with 1984 of
    /// those tokens already cached, for identical logits.
    ///
    /// Holds [`DEFAULT_PREFIX_CACHE_TOKENS`] tokens; use
    /// [`prefix_cache_tokens`](Self::prefix_cache_tokens) to choose. Prompts
    /// longer than that still work, they simply prefill from scratch.
    ///
    /// Off by default, for two reasons. The cache is eagerly allocated f32 KV, so
    /// it costs real memory: on Qwen2.5-0.5B, 24 layers of 2 KV heads, roughly
    /// 24KB per token, which is 96MB at the default and 768MB if sized to that
    /// model's full 32768-token context. And one cache holds one conversation, so
    /// generations on a single instance serialise instead of running concurrently.
    /// Ignored while speculative decoding is active.
    pub fn prefix_cache(mut self, enabled: bool) -> Self {
        self.prefix_cache_tokens = enabled.then_some(DEFAULT_PREFIX_CACHE_TOKENS);
        self
    }

    /// Enables prefix reuse with an explicit capacity in tokens.
    ///
    /// Capacity is the longest shared prefix that can be reused, and it is what
    /// the memory cost scales with: roughly 24KB per token on Qwen2.5-0.5B, more
    /// on models with more layers or KV heads. Clamped to the model's context.
    pub fn prefix_cache_tokens(mut self, tokens: usize) -> Self {
        self.prefix_cache_tokens = Some(tokens.max(1));
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.generation_overrides.temperature = Some(temp);
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max: usize) -> Self {
        self.generation_overrides.max_new_tokens = Some(max);
        self
    }

    /// Set top-p (nucleus) sampling threshold.
    pub fn top_p(mut self, p: f32) -> Self {
        self.generation_overrides.top_p = Some(p);
        self
    }

    /// Set top-k sampling limit.
    pub fn top_k(mut self, k: usize) -> Self {
        self.generation_overrides.top_k = Some(k);
        self
    }

    /// Set min-p sampling threshold.
    pub fn min_p(mut self, p: f32) -> Self {
        self.generation_overrides.min_p = Some(p);
        self
    }

    /// Set repetition penalty.
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.generation_overrides.repetition_penalty = Some(penalty);
        self
    }

    /// Use greedy decoding (temperature = 0, deterministic).
    pub fn greedy(mut self) -> Self {
        self.generation_overrides.temperature = Some(0.0);
        self.generation_overrides.do_sample = Some(false);
        self
    }

    /// Set all generation overrides at once.
    pub fn generation_config(mut self, overrides: GenerationOverrides) -> Self {
        self.generation_overrides = overrides;
        self
    }

    /// Configure model loading options.
    pub fn with_load_config<F>(mut self, f: F) -> Self
    where
        F: FnOnce(LoadConfigBuilder) -> LoadConfigBuilder,
    {
        self.load_config = Some(f(LoadConfigBuilder::new()).build());
        self
    }

    /// Set the cache directory for model files.
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Set the download policy.
    pub fn download_policy(mut self, policy: DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Never download models automatically.
    pub fn offline(mut self) -> Self {
        self.download_policy = DownloadPolicy::Never;
        self
    }

    /// Suppress informational output.
    pub fn quiet(mut self) -> Self {
        self.quiet = true;
        self
    }

    /// Allow suboptimal model choices without warnings.
    pub fn allow_warnings(mut self) -> Self {
        self.allow_warnings = true;
        self
    }

    /// Build the Generator.
    pub async fn build(self) -> GeneratorResult<Generator> {
        Generator::from_builder(self).await
    }
}
