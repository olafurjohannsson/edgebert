//! Model validation for chat compatibility.
//!
//! Provides tiered validation:
//! - Hard errors for capability mismatches (encoder used for chat)
//! - Warnings for suboptimal configurations (base model for chat)

use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};

use crate::chat::{ChatWarning, types::WarningSeverity};

use super::types::{ChatError, ChatResult};

/// Result of validating a model for chat use.
#[derive(Debug)]
pub struct ValidationResult {
    /// Model passed validation (may still have warnings).
    pub is_valid: bool,
    /// Warnings about suboptimal configuration.
    pub warnings: Vec<ChatWarning>,
}

impl ValidationResult {
    fn valid() -> Self {
        Self {
            is_valid: true,
            warnings: Vec::new(),
        }
    }

    fn valid_with_warning(warning: ChatWarning) -> Self {
        Self {
            is_valid: true,
            warnings: vec![warning],
        }
    }

    #[expect(
        dead_code,
        reason = "not referenced yet; kept until the path that needs it lands"
    )]
    fn invalid() -> Self {
        Self {
            is_valid: false,
            warnings: Vec::new(),
        }
    }
}

/// Validate that a model can be used for chat.
///
/// Returns Ok(ValidationResult) if the model can be used (possibly with warnings),
/// or Err(ChatError) if the model fundamentally cannot perform chat.
pub fn validate_for_chat(model_type: ModelType) -> ChatResult<ValidationResult> {
    let info = model_type.info();
    let cli_name = model_type.cli_name();

    match info.architecture {
        // Decoders can generate text - these are valid for chat
        ModelArchitecture::Llama
        | ModelArchitecture::Qwen2
        | ModelArchitecture::Mistral
        | ModelArchitecture::Phi3
        | ModelArchitecture::GPT => {
            // Valid architecture, continue to task check
        }

        // Encoders cannot generate text - hard error
        ModelArchitecture::Bert
        | ModelArchitecture::NomicBert
        | ModelArchitecture::Mpnet
        | ModelArchitecture::Clip => {
            return Err(ChatError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Model architecture '{}' is an encoder and cannot generate text. \
                     Use an Encoder for embeddings instead.",
                    info.architecture.display_name()
                ),
            });
        }

        ModelArchitecture::Whisper => {
            return Err(ChatError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Model architecture '{}' is designed for speech-to-text transcription. \
                     Use a SpeechToText model instead.",
                    info.architecture.display_name()
                ),
            });
        }

        // Encoder-decoders could theoretically work but are designed for different tasks
        ModelArchitecture::T5 | ModelArchitecture::Bart => {
            return Err(ChatError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Model architecture '{}' is a seq2seq model designed for translation/summarization. \
                     Use Translator or Summarizer instead.",
                    info.architecture.display_name()
                ),
            });
        }
    }

    match info.task {
        // Ideal: Model is designed for chat or reasoning
        ModelTask::Chat | ModelTask::Reasoning => Ok(ValidationResult::valid()),

        // Acceptable: Base generation models work but may not follow instructions well
        ModelTask::Generation => Ok(ValidationResult::valid_with_warning(ChatWarning {
            message: format!(
                "Model '{}' is a base model, not instruction-tuned.",
                cli_name
            ),
            severity: WarningSeverity::Info,
        })),

        // These shouldn't happen given the architecture check above, but be explicit
        ModelTask::Embedding
        | ModelTask::ImageEmbedding
        | ModelTask::ReRanking
        | ModelTask::Classification
        | ModelTask::SentimentAnalysis
        | ModelTask::ZeroShotClassification => Err(ChatError::IncompatibleModel {
            model: cli_name.to_string(),
            reason: format!(
                "Model is designed for {} tasks, not text generation.",
                format!("{:?}", info.task).to_lowercase()
            ),
        }),

        // Seq2seq tasks - shouldn't reach here due to architecture check
        ModelTask::Seq2Seq
        | ModelTask::Summarization
        | ModelTask::Translation
        | ModelTask::SpeechToText
        | ModelTask::TextToText => Err(ChatError::IncompatibleModel {
            model: cli_name.to_string(),
            reason: "Model is designed for seq2seq tasks like translation or summarization."
                .to_string(),
        }),
    }
}

/// Check if a model architecture supports text generation.
pub fn is_decoder_architecture(arch: ModelArchitecture) -> bool {
    matches!(
        arch,
        ModelArchitecture::Llama
            | ModelArchitecture::Qwen2
            | ModelArchitecture::Mistral
            | ModelArchitecture::Phi3
            | ModelArchitecture::GPT
    )
}

/// Check if a model is instruction-tuned.
pub fn is_instruct_model(model_type: ModelType) -> bool {
    model_type.is_instruct_model()
}

/// Get suggested models for chat, smallest first.
///
/// A curated shortlist rather than everything [`crate::chat::available_models`]
/// returns, since the point is to give someone a starting point rather than a
/// catalogue.
///
/// Written as [`ModelType`] variants, not string literals, on purpose. This list
/// used to be hardcoded names and four of its seven entries had gone stale —
/// `llama3.2-1b`, `llama3.2-3b`, `llama3.1-8b` and `qwen2.5-0.5b` all lost out to
/// an `-instruct` suffix at some point, so following a suggestion produced a
/// second "unknown model" error. Naming variants makes the compiler reject a
/// rename, and `cli_name()` keeps the strings in sync by construction.
pub fn suggest_chat_models() -> Vec<&'static str> {
    [
        ModelType::Qwen2_5_0_5B_Instruct,
        ModelType::Llama3_2_1B_Instruct,
        ModelType::Llama3_2_3B_Instruct,
        ModelType::Phi3_5_Mini_Instruct,
        ModelType::Mistral7B_v0_3_Instruct,
        ModelType::Llama3_1_8B_Instruct,
    ]
    .into_iter()
    .filter(|m| validate_for_chat(*m).is_ok())
    .map(|m| m.cli_name())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_valid_chat_models() {
    //     // These should all pass validation
    //     let valid_models = [
    //         "llama3.2-1b",
    //         "llama3.2-3b",
    //         "qwen2.5-0.5b",
    //         "qwen2.5-1.5b",
    //         "mistral-7b",
    //     ];

    //     for name in valid_models {
    //         let model_type = ModelType::from_cli_name(name).expect(name);
    //         let result = validate_for_chat(model_type);
    //         assert!(result.is_ok(), "Model {} should be valid for chat", name);
    //         assert!(
    //             result.unwrap().is_valid,
    //             "Model {} should pass validation",
    //             name
    //         );
    //     }
    // }

    #[test]
    fn test_invalid_encoder_models() {
        // Encoders should fail
        let encoder_models = ["minilm-l6-v2", "nomic-embed-text"];

        for name in encoder_models {
            let model_type = ModelType::from_cli_name(name).expect(name);
            let result = validate_for_chat(model_type);
            assert!(
                result.is_err(),
                "Encoder {} should fail chat validation",
                name
            );
        }
    }

    #[test]
    fn test_base_model_warning() {
        // Base models should pass with warning
        if let Some(model_type) = ModelType::from_cli_name("gpt2") {
            let result = validate_for_chat(model_type).expect("Should be valid");
            assert!(result.is_valid);
            assert!(
                !result.warnings.is_empty(),
                "Should have warning for base model"
            );
        }
    }
}
