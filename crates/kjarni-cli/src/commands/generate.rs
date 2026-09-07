//! Text generation command using decoder models

use anyhow::{Result, anyhow};
use futures::{StreamExt, pin_mut};

use kjarni::{
    DecoderGenerator, DecoderLanguageModel, DecodingStrategy, Device, GenerationConfig,
    ModelArchitecture, ModelType, SamplingParams, SpeculationParams, TokenType, WgpuContext,
    models::{Gpt2Model, LlamaModel, PhiModel, QwenModel},
    registry,
};
use std::io::{self, Write};
use std::sync::Arc;

use super::util::{model_not_found_error, resolve_input};

pub async fn run(
    prompt: Option<&str>,
    model: &str,
    model_path: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: f32,
    greedy: bool,
    gpu: bool,
    no_stream: bool,
    quiet: bool,
    draft: Option<&str>,
    draft_tokens: usize,
) -> Result<()> {
    // Resolve prompt
    let prompt_text = resolve_input(prompt)?;

    // Resolve model
    let device = if gpu { Device::Wgpu } else { Device::Cpu };
    if gpu && let Some(mt) = ModelType::from_cli_name(model) {
        super::util::check_gpu_capacity(mt)?;
    }

    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| anyhow!(model_not_found_error(model, Some("decoder"))))?;

    if !is_supported_decoder_architecture(model_type.architecture()) {
        return Err(anyhow!(
            "Model '{}' is not a decoder. Use a decoder model for generation. Detected architecture: {:?}",
            model,
            model_type.architecture()
        ));
    }

    // Check if downloaded
    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("Model '{}' not found locally. Downloading...", model);
        }
        registry::download_model(model, false, quiet).await?;
        if !quiet {
            eprintln!();
        }
    }

    // Load model
    if !quiet {
        eprintln!("Loading model '{}'...", model);
    }

    // A local path, which may be a `.gguf` file or a safetensors directory.
    //
    // `--model` still supplies the architecture, since a path on its own does not
    // say which family the weights belong to. This is the only way to reach a
    // quantised model: the registry has no GGUF entries, so `llama3.2-3b-instruct`
    // resolves to the 6GB bf16 copy and decode streams four times the weights a
    // Q4_K_M file would.
    let loaded_model: Arc<dyn DecoderLanguageModel> = if let Some(path) = model_path {
        let p = std::path::Path::new(path);
        if !p.exists() {
            return Err(anyhow!("--model-path not found: {path}"));
        }
        if !model_type.is_llama_model() {
            return Err(anyhow!(
                "--model-path currently supports Llama-family weights only. \
                 Pass --model with a Llama architecture, or omit --model-path."
            ));
        }
        if !quiet {
            eprintln!("Loading weights from {path}...");
        }
        // `load_from_pretrained` is sync and cannot build a context itself, unlike
        // the registry path which is async and does. Without one the GPU RoPE
        // kernel has nothing to run on.
        let context = if device.is_gpu() {
            Some(WgpuContext::new().await?)
        } else {
            None
        };
        Arc::new(LlamaModel::from_pretrained(
            p,
            device,
            context,
            None,
            Some(model_type),
        )?)
    } else if model_type.is_llama_model() {
        Arc::new(LlamaModel::from_registry(model_type, None, device, None, None).await?)
    } else if model_type.is_qwen_model() {
        // The registry has carried `is_qwen_model` all along and this dispatch
        // never used it, so `kjarni chat` ran Qwen while `kjarni generate`
        // rejected it as unsupported.
        Arc::new(QwenModel::from_registry(model_type, None, device, None, None).await?)
    } else if model_type.is_phi_model() {
        Arc::new(PhiModel::from_registry(model_type, None, device, None, None).await?)
    } else if model_type.is_gpt2_model() {
        Arc::new(Gpt2Model::from_registry(model_type, None, device, None, None).await?)
    } else {
        return Err(anyhow!(
            "Model '{}' not yet supported for generation.",
            model
        ));
    };

    let mut generator = DecoderGenerator::new(loaded_model)?;

    // A draft model proposes several tokens and the target verifies them in one
    // pass. Decode is bandwidth bound, so reading the target's weights once for
    // k tokens instead of k times is the whole saving. The draft must share the
    // target's vocabulary or its proposals cannot be scored.
    let speculation = if let Some(draft_name) = draft {
        let draft_type = ModelType::from_cli_name(draft_name)
            .ok_or_else(|| anyhow!("{}", model_not_found_error(draft_name, Some("decoder"))))?;
        if !quiet {
            eprintln!("Loading draft model '{draft_name}'...");
        }
        let draft_model = load_decoder_from_registry(draft_type, device).await?;
        generator.load_draft_model(draft_model)?;
        Some(SpeculationParams {
            num_tokens: draft_tokens,
            probabilistic: !greedy,
        })
    } else {
        None
    };

    // Configure generation
    let config = build_generation_config(
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        greedy,
        speculation,
    );

    if !quiet {
        eprintln!();
    }

    // Generate
    if no_stream {
        let output = generator.generate(&prompt_text, &config, None).await?;
        println!("{}", output);
    } else {
        let stream = generator
            .generate_stream(&prompt_text, &config, None)
            .await?;
        pin_mut!(stream);

        let mut stdout = io::stdout();
        let mut generated_any = false;

        while let Some(token_result) = stream.next().await {
            let token = token_result?;

            // Skip prompt tokens
            if token.token_type == TokenType::Prompt {
                continue;
            }

            print!("{}", token.text);
            stdout.flush()?;
            generated_any = true;
        }

        if generated_any {
            println!();
        }
    }

    Ok(())
}

/// Check if the architecture is a supported decoder for generation
/// Whether an architecture can generate text.
///
/// This asks the registry rather than keeping a second list. The hardcoded copy that
/// used to live here had drifted: it omitted Phi3, so `generate --model phi3.5-mini`
/// refused a model the engine implements and the registry already classifies as a
/// decoder. Any list maintained in parallel with the registry will drift again.
fn is_supported_decoder_architecture(arch: ModelArchitecture) -> bool {
    arch.category() == "decoder"
}

/// Loads a decoder from the registry, dispatching on architecture.
///
/// The same chain the target model uses, factored out so a draft model can be
/// loaded the same way. Registry only: a draft is named, not given as a path.
async fn load_decoder_from_registry(
    model_type: ModelType,
    device: Device,
) -> Result<Arc<dyn DecoderLanguageModel>> {
    let m: Arc<dyn DecoderLanguageModel> = if model_type.is_llama_model() {
        Arc::new(LlamaModel::from_registry(model_type, None, device, None, None).await?)
    } else if model_type.is_qwen_model() {
        Arc::new(QwenModel::from_registry(model_type, None, device, None, None).await?)
    } else if model_type.is_phi_model() {
        Arc::new(PhiModel::from_registry(model_type, None, device, None, None).await?)
    } else if model_type.is_gpt2_model() {
        Arc::new(Gpt2Model::from_registry(model_type, None, device, None, None).await?)
    } else {
        return Err(anyhow!(
            "Model '{}' not yet supported for generation.",
            model_type.cli_name()
        ));
    };
    Ok(m)
}

/// Build the decoding strategy based on parameters
fn build_decoding_strategy(
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    greedy: bool,
) -> DecodingStrategy {
    if greedy || temperature == 0.0 {
        DecodingStrategy::Greedy
    } else {
        DecodingStrategy::Sample(SamplingParams {
            temperature,
            top_k: top_k.or(Some(50)),
            top_p: top_p.or(Some(0.9)),
            min_p: min_p.or(Some(0.1)),
        })
    }
}

/// Build the full generation config
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the CLI flags one to one"
)]
fn build_generation_config(
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: f32,
    greedy: bool,
    speculation: Option<SpeculationParams>,
) -> GenerationConfig {
    let strategy = build_decoding_strategy(temperature, top_k, top_p, min_p, greedy);

    GenerationConfig {
        max_new_tokens: Some(max_tokens),
        repetition_penalty,
        strategy,
        speculation,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_supported_gpt() {
        assert!(is_supported_decoder_architecture(ModelArchitecture::GPT));
    }

    #[test]
    fn test_supported_llama() {
        assert!(is_supported_decoder_architecture(ModelArchitecture::Llama));
    }

    #[test]
    fn test_supported_mistral() {
        assert!(is_supported_decoder_architecture(
            ModelArchitecture::Mistral
        ));
    }

    #[test]
    fn test_supported_qwen2() {
        assert!(is_supported_decoder_architecture(ModelArchitecture::Qwen2));
    }

    #[test]
    fn test_supported_phi3() {
        assert!(is_supported_decoder_architecture(ModelArchitecture::Phi3));
    }

    /// The cases above are a list maintained in parallel with the registry, which
    /// is the drift that made `generate` refuse Phi3 in the first place. This one
    /// has no wildcard arm, so adding an architecture stops compiling here until
    /// someone decides whether `generate` should accept it.
    #[test]
    fn test_every_architecture_is_classified() {
        let all = [
            ModelArchitecture::Llama,
            ModelArchitecture::Qwen2,
            ModelArchitecture::Mistral,
            ModelArchitecture::Phi3,
            ModelArchitecture::GPT,
            ModelArchitecture::Bert,
            ModelArchitecture::NomicBert,
            ModelArchitecture::Mpnet,
            ModelArchitecture::T5,
            ModelArchitecture::Bart,
            ModelArchitecture::Whisper,
            ModelArchitecture::Clip,
        ];

        for arch in all {
            let expected = match arch {
                ModelArchitecture::Llama
                | ModelArchitecture::Qwen2
                | ModelArchitecture::Mistral
                | ModelArchitecture::Phi3
                | ModelArchitecture::GPT => true,
                ModelArchitecture::Bert
                | ModelArchitecture::NomicBert
                | ModelArchitecture::Mpnet
                | ModelArchitecture::T5
                | ModelArchitecture::Bart
                | ModelArchitecture::Whisper
                | ModelArchitecture::Clip => false,
            };

            assert_eq!(
                is_supported_decoder_architecture(arch),
                expected,
                "{} is classified as {:?} by the registry",
                arch.display_name(),
                arch.category()
            );
        }
    }

    #[test]
    fn test_unsupported_bert() {
        assert!(!is_supported_decoder_architecture(ModelArchitecture::Bert));
    }

    #[test]
    fn test_unsupported_t5() {
        assert!(!is_supported_decoder_architecture(ModelArchitecture::T5));
    }

    #[test]
    fn test_unsupported_bart() {
        assert!(!is_supported_decoder_architecture(ModelArchitecture::Bart));
    }

    #[test]
    fn test_unsupported_whisper() {
        assert!(!is_supported_decoder_architecture(
            ModelArchitecture::Whisper
        ));
    }

    #[test]
    fn test_unsupported_nomic_bert() {
        assert!(!is_supported_decoder_architecture(
            ModelArchitecture::NomicBert
        ));
    }

    #[test]
    fn test_strategy_greedy_flag() {
        let strategy = build_decoding_strategy(0.7, None, None, None, true);
        assert!(matches!(strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_strategy_zero_temperature() {
        let strategy = build_decoding_strategy(0.0, None, None, None, false);
        assert!(matches!(strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_strategy_sampling_default() {
        let strategy = build_decoding_strategy(0.7, None, None, None, false);

        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.7);
                assert_eq!(params.top_k, Some(50));
                assert_eq!(params.top_p, Some(0.9));
                assert_eq!(params.min_p, Some(0.1));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_strategy_sampling_custom_top_k() {
        let strategy = build_decoding_strategy(0.7, Some(100), None, None, false);

        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.top_k, Some(100));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_strategy_sampling_custom_top_p() {
        let strategy = build_decoding_strategy(0.7, None, Some(0.95), None, false);

        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.top_p, Some(0.95));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_strategy_sampling_custom_min_p() {
        let strategy = build_decoding_strategy(0.7, None, None, Some(0.05), false);

        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.min_p, Some(0.05));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_strategy_sampling_all_custom() {
        let strategy = build_decoding_strategy(1.0, Some(40), Some(0.8), Some(0.02), false);

        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 1.0);
                assert_eq!(params.top_k, Some(40));
                assert_eq!(params.top_p, Some(0.8));
                assert_eq!(params.min_p, Some(0.02));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_strategy_greedy_overrides_temperature() {
        let strategy = build_decoding_strategy(1.5, Some(100), Some(0.99), Some(0.01), true);
        assert!(matches!(strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_generation_config_basic() {
        let config = build_generation_config(100, 0.7, None, None, None, 1.1, false, None);

        assert_eq!(config.max_new_tokens, Some(100));
        assert_eq!(config.repetition_penalty, 1.1);
        assert!(matches!(config.strategy, DecodingStrategy::Sample(_)));
    }

    #[test]
    fn test_generation_config_greedy() {
        let config = build_generation_config(50, 0.7, None, None, None, 1.0, true, None);

        assert_eq!(config.max_new_tokens, Some(50));
        assert!(matches!(config.strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_generation_config_zero_temp_greedy() {
        let config = build_generation_config(200, 0.0, None, None, None, 1.2, false, None);

        assert!(matches!(config.strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_generation_config_max_tokens() {
        let config = build_generation_config(1000, 0.7, None, None, None, 1.0, false, None);
        assert_eq!(config.max_new_tokens, Some(1000));
    }

    #[test]
    fn test_generation_config_repetition_penalty() {
        let config = build_generation_config(100, 0.7, None, None, None, 1.5, false, None);
        assert_eq!(config.repetition_penalty, 1.5);
    }

    #[test]
    fn test_generation_config_no_repetition_penalty() {
        let config = build_generation_config(100, 0.7, None, None, None, 1.0, false, None);
        assert_eq!(config.repetition_penalty, 1.0);
    }

    #[test]
    fn test_generation_config_sampling_params_passed_through() {
        let config =
            build_generation_config(100, 0.9, Some(40), Some(0.85), Some(0.05), 1.1, false, None);

        match config.strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.9);
                assert_eq!(params.top_k, Some(40));
                assert_eq!(params.top_p, Some(0.85));
                assert_eq!(params.min_p, Some(0.05));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }
    #[test]
    fn test_very_low_temperature() {
        let strategy = build_decoding_strategy(0.001, None, None, None, false);

        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.001);
            }
            _ => panic!("Expected Sample strategy (temperature > 0)"),
        }
    }

    #[test]
    fn test_very_high_temperature() {
        let strategy = build_decoding_strategy(5.0, None, None, None, false);

        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 5.0);
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_large_max_tokens() {
        let config = build_generation_config(100_000, 0.7, None, None, None, 1.0, false, None);
        assert_eq!(config.max_new_tokens, Some(100_000));
    }

    #[test]
    fn test_small_max_tokens() {
        let config = build_generation_config(1, 0.7, None, None, None, 1.0, false, None);
        assert_eq!(config.max_new_tokens, Some(1));
    }

    #[test]
    fn test_zero_max_tokens() {
        let config = build_generation_config(0, 0.7, None, None, None, 1.0, false, None);
        assert_eq!(config.max_new_tokens, Some(0));
    }

    #[test]
    fn test_negative_repetition_penalty() {
        // Unusual but should be accepted by the config builder
        let config = build_generation_config(100, 0.7, None, None, None, -0.5, false, None);
        assert_eq!(config.repetition_penalty, -0.5);
    }

    #[test]
    fn test_high_repetition_penalty() {
        let config = build_generation_config(100, 0.7, None, None, None, 5.0, false, None);
        assert_eq!(config.repetition_penalty, 5.0);
    }
    #[test]
    fn test_typical_creative_writing_params() {
        let config =
            build_generation_config(512, 1.0, None, Some(0.95), Some(0.05), 1.1, false, None);

        match config.strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 1.0);
                assert_eq!(params.top_p, Some(0.95));
                assert_eq!(params.min_p, Some(0.05));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_typical_code_generation_params() {
        // Lower temperature, stricter sampling
        let config = build_generation_config(256, 0.3, Some(40), Some(0.9), None, 1.0, false, None);

        match config.strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.3);
                assert_eq!(params.top_k, Some(40));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_deterministic_generation_params() {
        // Greedy for reproducibility
        let config = build_generation_config(100, 0.0, None, None, None, 1.0, false, None);
        assert!(matches!(config.strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_chatbot_params() {
        // Balanced settings
        let config =
            build_generation_config(512, 0.7, Some(50), Some(0.9), Some(0.1), 1.1, false, None);

        assert_eq!(config.max_new_tokens, Some(512));
        assert_eq!(config.repetition_penalty, 1.1);

        match config.strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.7);
                assert_eq!(params.top_k, Some(50));
                assert_eq!(params.top_p, Some(0.9));
                assert_eq!(params.min_p, Some(0.1));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }
}
