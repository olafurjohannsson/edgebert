//! Builder pattern for Chat configuration.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_transformers::WgpuContext;

use crate::chat::presets::ChatPreset;
use crate::common::{DownloadPolicy, KjarniDevice, LoadConfig, LoadConfigBuilder};
use crate::generation::GenerationOverrides;
use crate::generator::DEFAULT_PREFIX_CACHE_TOKENS;

use super::model::Chat;
use super::types::{ChatMode, ChatResult};

/// Builder for configuring a Chat instance.
///
/// # Example
///
/// ```ignore
/// let chat = Chat::builder("llama3.2-1b-instruct")
///     .system("You are a helpful assistant.")
///     .mode(ChatMode::Creative)
///     .temperature(0.8)
///     .build()
///     .await?;
/// ```
pub struct ChatBuilder {
    // Model selection
    pub(crate) model: String,
    #[expect(
        dead_code,
        reason = "not referenced yet; kept until the path that needs it lands"
    )]
    pub(crate) model_path: Option<PathBuf>,

    // Chat-specific
    pub(crate) system_prompt: Option<String>,
    pub(crate) mode: ChatMode,

    // Device configuration
    pub(crate) device: KjarniDevice,
    pub(crate) context: Option<Arc<WgpuContext>>,

    // Model loading
    pub(crate) cache_dir: Option<PathBuf>,
    pub(crate) download_policy: DownloadPolicy,
    pub(crate) load_config: Option<LoadConfig>,

    // Generation defaults
    /// Draft model for speculative decoding, and how many tokens it proposes.
    pub(crate) draft_model: Option<String>,
    pub(crate) draft_tokens: usize,
    pub(crate) prefix_cache_tokens: Option<usize>,

    pub(crate) generation_overrides: GenerationOverrides,

    // Behavior
    pub(crate) quiet: bool,
    pub(crate) allow_suboptimal: bool,
}

impl ChatBuilder {
    /// Create a new builder for the specified model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_path: None,
            system_prompt: None,
            mode: ChatMode::default(),
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
            allow_suboptimal: false,
        }
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &ChatPreset) -> Self {
        let mut builder = Self::new(preset.model);
        builder.device = preset.recommended_device;
        builder.mode = preset.mode;

        if let Some(system) = preset.system_prompt {
            builder.system_prompt = Some(system.to_string());
        }

        if let Some(temp) = preset.temperature {
            builder.generation_overrides.temperature = Some(temp);
        }

        if let Some(max) = preset.max_tokens {
            builder.generation_overrides.max_new_tokens = Some(max);
        }

        builder
    }

    /// Set the system prompt.
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the chat mode.
    pub fn mode(mut self, mode: ChatMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn creative(self) -> Self {
        self.mode(ChatMode::Creative)
    }

    pub fn reasoning(self) -> Self {
        self.mode(ChatMode::Reasoning)
    }
    pub fn device(mut self, device: KjarniDevice) -> Self {
        self.device = device;
        self
    }

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

    /// Set the sampling temperature.
    /// Enables speculative decoding with `name` as the draft model.
    ///
    /// The draft must share this model's vocabulary — a smaller model of the same
    /// family. Measured on Qwen2.5, 0.5B drafting for 1.5B is 3.5x faster at
    /// eight proposed tokens, because verifying a batch reads the target's
    /// weights once instead of once per token.
    pub fn draft(mut self, name: impl Into<String>, num_tokens: usize) -> Self {
        self.draft_model = Some(name.into());
        self.draft_tokens = num_tokens.max(1);
        self
    }

    /// Reuses the KV cache across turns, so a conversation prefills only the new
    /// message instead of replaying its whole history.
    ///
    /// Every turn re-sends the transcript, so turn N normally pays to prefill
    /// everything said before it. With this on, only the tail is new: on
    /// Qwen2.5-0.5B a 2048 token prompt costs 15.96s cold and 0.49s with 1984
    /// tokens already cached, for identical logits.
    ///
    /// Holds [`DEFAULT_PREFIX_CACHE_TOKENS`] tokens; use
    /// [`prefix_cache_tokens`](Self::prefix_cache_tokens) to choose. Longer
    /// conversations still work, they simply stop reusing once they outgrow it.
    ///
    /// Off by default: the KV cache is eagerly allocated f32, roughly 24KB per
    /// token on Qwen2.5-0.5B, and one cache holds one conversation, so sends on a
    /// single instance serialise. Ignored while speculative decoding is active.
    pub fn prefix_cache(mut self, enabled: bool) -> Self {
        self.prefix_cache_tokens = enabled.then_some(DEFAULT_PREFIX_CACHE_TOKENS);
        self
    }

    /// Enables prefix reuse with an explicit capacity in tokens.
    ///
    /// Capacity is the longest transcript that can be reused, and what the memory
    /// cost scales with. Clamped to the model's context.
    pub fn prefix_cache_tokens(mut self, tokens: usize) -> Self {
        self.prefix_cache_tokens = Some(tokens.max(1));
        self
    }

    /// Use greedy decoding (temperature = 0, deterministic).
    ///
    /// Setting `temperature(0.0)` alone is not enough: without `do_sample = false`
    /// the sampler still runs, and two identical calls can differ.
    pub fn greedy(mut self) -> Self {
        self.generation_overrides.temperature = Some(0.0);
        self.generation_overrides.do_sample = Some(false);
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
    pub fn allow_suboptimal(mut self) -> Self {
        self.allow_suboptimal = true;
        self
    }

    /// Build the Chat instance.
    pub async fn build(self) -> ChatResult<Chat> {
        Chat::from_builder(self).await
    }
}

impl Chat {
    /// Create a builder from a preset.
    pub fn from_preset(preset: &ChatPreset) -> ChatBuilder {
        ChatBuilder::from_preset(preset)
    }
}
