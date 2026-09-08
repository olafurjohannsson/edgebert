//! Model validation for text generation.

use kjarni_transformers::models::{ModelArchitecture, ModelType};

use super::types::{GeneratorError, GeneratorResult};

/// Validate that a model can be used for text generation.
pub fn validate_for_generation(model_type: ModelType) -> GeneratorResult<()> {
    let info = model_type.info();
    let cli_name = model_type.cli_name();

    match info.architecture {
        // Decoder-only models can generate text
        ModelArchitecture::Llama
        | ModelArchitecture::Qwen2
        | ModelArchitecture::Mistral
        | ModelArchitecture::Phi3
        | ModelArchitecture::GPT => {
            // Valid architecture
        }

        // Encoders cannot generate text
        ModelArchitecture::Bert
        | ModelArchitecture::NomicBert
        | ModelArchitecture::Mpnet
        | ModelArchitecture::Clip => {
            return Err(GeneratorError::InvalidModel(
                cli_name.to_string(),
                format!(
                    "Architecture '{}' is an encoder and cannot generate text. Use Embedder instead.",
                    info.architecture.display_name()
                ),
            ));
        }

        // Whisper is for speech
        ModelArchitecture::Whisper => {
            return Err(GeneratorError::InvalidModel(
                cli_name.to_string(),
                "Whisper is designed for speech-to-text. Use Transcriber instead.".to_string(),
            ));
        }

        // Seq2seq models should use Seq2SeqGenerator
        ModelArchitecture::T5 | ModelArchitecture::Bart => {
            return Err(GeneratorError::InvalidModel(
                cli_name.to_string(),
                format!(
                    "Architecture '{}' is a seq2seq model. Use Seq2SeqGenerator, Translator, or Summarizer instead.",
                    info.architecture.display_name()
                ),
            ));
        }
    }

    Ok(())
}

/// Get all available models for text generation.
pub fn get_generator_models() -> Vec<&'static str> {
    ModelType::all()
        .filter(|m| validate_for_generation(*m).is_ok())
        .map(|m| m.cli_name())
        .collect()
}

/// Get suggested models for text generation.
pub fn suggest_generator_models() -> Vec<&'static str> {
    vec![
        "qwen2.5-0.5b-instruct",
        "qwen2.5-1.5b-instruct",
        "llama3.2-1b-instruct",
        "llama3.2-3b-instruct",
        "mistral-7b-instruct",
        "phi3.5-mini-instruct",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_decoder_models() {
        let valid_models = ["llama3.2-1b-instruct", "qwen2.5-0.5b-instruct"];

        for name in valid_models {
            if let Some(model_type) = ModelType::from_cli_name(name) {
                assert!(
                    validate_for_generation(model_type).is_ok(),
                    "Model {} should be valid",
                    name
                );
            }
        }
    }

    #[test]
    fn test_validate_encoder_rejected() {
        let encoder_models = ["minilm-l6-v2", "nomic-embed-text"];

        for name in encoder_models {
            if let Some(model_type) = ModelType::from_cli_name(name) {
                assert!(
                    validate_for_generation(model_type).is_err(),
                    "Encoder {} should be rejected",
                    name
                );
            }
        }
    }

    #[test]
    fn test_get_generator_models() {
        let models = get_generator_models();
        assert!(!models.is_empty());

        // Should contain LLM models
        let has_llm = models.iter().any(|m| {
            m.contains("llama") || m.contains("qwen") || m.contains("mistral") || m.contains("phi")
        });
        assert!(has_llm, "Should contain LLM models");

        // Should NOT contain encoders
        assert!(!models.contains(&"minilm-l6-v2"));
        assert!(!models.contains(&"nomic-embed-text"));
    }

    #[test]
    fn test_suggest_generator_models() {
        let models = suggest_generator_models();
        assert!(!models.is_empty());
    }
}
