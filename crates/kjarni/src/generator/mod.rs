//! Raw text generator for decoder language models.

mod builder;
mod model;
pub mod presets;
mod types;
mod validation;

pub use builder::{DEFAULT_PREFIX_CACHE_TOKENS, GeneratorBuilder};
use kjarni_transformers::ModelType;
pub use model::Generator;
pub use types::*;

/// Generate text with default settings.
pub async fn generate(model: &str, prompt: &str) -> GeneratorResult<String> {
    Generator::new(model).await?.generate(prompt).await
}

/// List available generation models.
pub fn available_models() -> Vec<&'static str> {
    validation::get_generator_models()
}

/// Get suggested models for generation.
pub fn suggested_models() -> Vec<&'static str> {
    validation::suggest_generator_models()
}

/// Check if a model can be used for generation.
pub fn is_generator_model(model: &str) -> GeneratorResult<()> {
    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| GeneratorError::UnknownModel(model.to_string()))?;
    validation::validate_for_generation(model_type)
}

#[cfg(test)]
mod tests;
