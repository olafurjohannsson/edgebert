//! Core Generator implementation.

use std::sync::Arc;

use anyhow::anyhow;
use futures::{Stream, StreamExt, pin_mut};
use tokio::sync::mpsc;

use kjarni_transformers::WgpuContext;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::models::{ModelArchitecture, ModelType};
use kjarni_transformers::traits::Device;

use crate::common::DownloadPolicy;
use crate::generation::{GenerationOverrides, ResolvedGenerationConfig, resolve_generation_config};
use crate::{DecoderGenerator, DecoderLanguageModel, TokenType};

use super::builder::GeneratorBuilder;
use super::types::{GeneratedToken, GeneratorError, GeneratorResult};
use super::validation::validate_for_generation;
use kjarni_transformers::models::get_default_cache_dir;

/// Raw text generator for decoder language models.
///
/// Provides direct access to language model text generation without chat
/// templates, system prompts, or conversation management.
pub struct Generator {
    pub(crate) decoder: Arc<DecoderGenerator>,
    model_type: ModelType,
    generation_config: ResolvedGenerationConfig,
    user_overrides: GenerationOverrides,
    device: Device,
    context: Option<Arc<WgpuContext>>,
}

impl Generator {
    /// Creates a Generator with default settings.
    pub async fn new(model: &str) -> GeneratorResult<Self> {
        GeneratorBuilder::new(model).build().await
    }

    /// Creates a builder for custom configuration.
    pub fn builder(model: &str) -> GeneratorBuilder {
        GeneratorBuilder::new(model)
    }

    pub(crate) async fn from_builder(builder: GeneratorBuilder) -> GeneratorResult<Self> {
        let model_type = ModelType::from_cli_name(&builder.model)
            .ok_or_else(|| GeneratorError::UnknownModel(builder.model.clone()))?;

        validate_for_generation(model_type)?;

        // if !builder.quiet && !builder.allow_warnings {
        //     for warning in &validation.warnings {
        //         eprintln!("warning: {}", warning);
        //     }
        // }

        let cache_dir = builder
            .cache_dir
            .clone()
            .unwrap_or_else(get_default_cache_dir);

        let is_downloaded = model_type.is_downloaded(&cache_dir);

        if !is_downloaded {
            match builder.download_policy {
                DownloadPolicy::Never => {
                    return Err(GeneratorError::ModelNotDownloaded(builder.model.clone()));
                }
                DownloadPolicy::IfMissing | DownloadPolicy::Eager => {
                    if !builder.quiet {
                        eprintln!("downloading model '{}'...", builder.model);
                    }
                    kjarni_transformers::models::download_model_files(
                        &model_type.cache_dir(&cache_dir),
                        &model_type.info().paths,
                        kjarni_transformers::models::registry::WeightsFormat::SafeTensors,
                        builder.quiet,
                    )
                    .await
                    .map_err(|e| GeneratorError::DownloadFailed {
                        model: builder.model.clone(),
                        source: e,
                    })?;
                }
            }
        }

        let device = builder.device.to_device();

        let context = if device == Device::Wgpu {
            if let Some(ctx) = builder.context {
                Some(ctx)
            } else {
                Some(
                    WgpuContext::new()
                        .await
                        .map_err(|_| GeneratorError::GpuUnavailable)?,
                )
            }
        } else {
            None
        };

        let load_config = builder
            .load_config
            .map(|c| c.into_inner())
            .unwrap_or_default();

        let model: Arc<dyn DecoderLanguageModel> =
            Self::load_model(model_type, &cache_dir, device, context.clone(), load_config)
                .await
                .map_err(|e| GeneratorError::LoadFailed {
                    model: builder.model.clone(),
                    source: e,
                })?;

        let mut decoder_inner =
            DecoderGenerator::new(model.clone()).map_err(|e| GeneratorError::LoadFailed {
                model: builder.model.clone(),
                source: e,
            })?;

        // The draft is loaded before the generator is shared, because attaching
        // one needs &mut. It must share the target's vocabulary; a mismatch is
        // reported here rather than producing proposals that can never be scored.
        let speculation = if let Some(ref draft_name) = builder.draft_model {
            let draft_type = ModelType::from_cli_name(draft_name)
                .ok_or_else(|| GeneratorError::UnknownModel(draft_name.clone()))?;
            let draft_model = Self::load_model(
                draft_type,
                &cache_dir,
                device,
                context.clone(),
                Default::default(),
            )
            .await
            .map_err(|e| GeneratorError::LoadFailed {
                model: draft_name.clone(),
                source: e,
            })?;
            decoder_inner.load_draft_model(draft_model).map_err(|e| {
                GeneratorError::LoadFailed {
                    model: draft_name.clone(),
                    source: e,
                }
            })?;
            Some(kjarni_transformers::common::SpeculationParams {
                num_tokens: builder.draft_tokens,
                probabilistic: true,
            })
        } else {
            None
        };

        // Capacity caps the prefix that can be reused, and the KV cache behind it
        // is allocated eagerly, so this is a memory decision the caller makes
        // rather than the model's full context window.
        if let Some(tokens) = builder.prefix_cache_tokens {
            decoder_inner
                .enable_prefix_cache(tokens)
                .map_err(|e| GeneratorError::LoadFailed {
                    model: builder.model.clone(),
                    source: e,
                })?;
        }

        let decoder = Arc::new(decoder_inner);

        let model_defaults = model.get_default_generation_config();
        let mut generation_config = resolve_generation_config(
            model_defaults,
            &builder.generation_overrides,
            &GenerationOverrides::default(),
        );
        generation_config.inner.speculation = speculation;

        Ok(Self {
            decoder,
            model_type,
            generation_config,
            user_overrides: builder.generation_overrides,
            device,
            context,
        })
    }

    async fn load_model(
        model_type: ModelType,
        cache_dir: &std::path::Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: ModelLoadConfig,
    ) -> anyhow::Result<Arc<dyn DecoderLanguageModel + Send + Sync>> {
        let info = model_type.info();

        match info.architecture {
            ModelArchitecture::Llama => {
                use kjarni_models::models::llama::LlamaModel;
                let model = LlamaModel::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }

            ModelArchitecture::Qwen2 => {
                use kjarni_models::models::qwen::QwenModel;
                let model = QwenModel::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }

            ModelArchitecture::Mistral => {
                use kjarni_models::models::mistral::MistralModel;
                let model = MistralModel::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }

            ModelArchitecture::GPT => {
                use kjarni_models::models::gpt2::Gpt2Model;
                let model = Gpt2Model::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }
            ModelArchitecture::Phi3 => {
                use kjarni_models::models::phi::PhiModel;
                let model = PhiModel::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }
            _ => Err(anyhow!(
                "architecture {:?} is not supported for text generation",
                info.architecture
            )),
        }
    }

    /// Generates text from a prompt.
    pub async fn generate(&self, prompt: &str) -> GeneratorResult<String> {
        self.generate_with_config(prompt, &GenerationOverrides::default())
            .await
    }

    /// Generates with custom overrides for this call only.
    pub async fn generate_with_config(
        &self,
        prompt: &str,
        runtime_overrides: &GenerationOverrides,
    ) -> GeneratorResult<String> {
        let config = resolve_generation_config(
            self.generation_config.inner.clone(),
            &self.user_overrides,
            runtime_overrides,
        );

        let stream = self
            .decoder
            .generate_stream(prompt, config.as_ref(), None)
            .await?;

        pin_mut!(stream);

        let mut output = String::new();
        while let Some(token_result) = stream.next().await {
            let token = token_result?;

            if token.token_type == TokenType::Prompt {
                continue;
            }

            output.push_str(&token.text);
        }

        Ok(output)
    }

    /// Streams generated tokens.
    pub async fn stream(
        &self,
        prompt: &str,
    ) -> GeneratorResult<
        std::pin::Pin<Box<dyn Stream<Item = GeneratorResult<GeneratedToken>> + Send>>,
    > {
        self.stream_with_config(prompt, GenerationOverrides::default())
            .await
    }

    /// Streams with custom overrides.
    pub async fn stream_with_config(
        &self,
        prompt: &str,
        runtime_overrides: GenerationOverrides,
    ) -> GeneratorResult<
        std::pin::Pin<Box<dyn Stream<Item = GeneratorResult<GeneratedToken>> + Send>>,
    > {
        let config = resolve_generation_config(
            self.generation_config.inner.clone(),
            &self.user_overrides,
            &runtime_overrides,
        );

        let decoder = self.decoder.clone();
        let config = config.into_inner();
        let prompt = prompt.to_string();

        let (tx, rx) = mpsc::channel::<GeneratorResult<GeneratedToken>>(32);

        tokio::spawn(async move {
            let stream = match decoder.generate_stream(&prompt, &config, None).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(GeneratorError::GenerationFailed(e))).await;
                    return;
                }
            };

            pin_mut!(stream);

            while let Some(result) = stream.next().await {
                let msg = match result {
                    Ok(token) => {
                        if token.token_type == TokenType::Prompt {
                            continue;
                        }

                        Ok(GeneratedToken {
                            text: token.text,
                            id: token.id,
                            is_special: token.token_type == TokenType::Special,
                        })
                    }
                    Err(e) => Err(GeneratorError::GenerationFailed(e)),
                };

                if tx.send(msg).await.is_err() {
                    break;
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    /// Streams as simple strings.
    pub async fn stream_text(
        &self,
        prompt: &str,
    ) -> GeneratorResult<std::pin::Pin<Box<dyn Stream<Item = GeneratorResult<String>> + Send>>>
    {
        let inner_stream = self.stream(prompt).await?;
        let mapped = inner_stream.map(|result| result.map(|token| token.text));
        Ok(Box::pin(mapped))
    }

    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    pub fn model_name(&self) -> &str {
        self.model_type.cli_name()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn context_size(&self) -> usize {
        self.decoder.model.context_size()
    }

    pub fn vocab_size(&self) -> usize {
        self.decoder.model.vocab_size()
    }

    pub fn generation_config(&self) -> &ResolvedGenerationConfig {
        &self.generation_config
    }

    /// Returns a reference to the underlying decoder generator.
    pub fn decoder(&self) -> &DecoderGenerator {
        &self.decoder
    }

    pub fn gpu_context(&self) -> Option<&Arc<WgpuContext>> {
        self.context.as_ref()
    }
}

#[allow(
    dead_code,
    reason = "not referenced yet; kept until the path that needs it lands"
)]
/// Generates text with default settings.
pub async fn generate(model: &str, prompt: &str) -> GeneratorResult<String> {
    Generator::new(model).await?.generate(prompt).await
}

#[expect(
    dead_code,
    reason = "not referenced yet; kept until the path that needs it lands"
)]
/// Generates text with custom configuration.
pub async fn generate_with_config(
    model: &str,
    prompt: &str,
    config: GenerationOverrides,
) -> GeneratorResult<String> {
    Generator::builder(model)
        .generation_config(config)
        .build()
        .await?
        .generate(prompt)
        .await
}
