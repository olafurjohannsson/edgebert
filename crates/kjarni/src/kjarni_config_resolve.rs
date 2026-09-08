use kjarni_transformers::{models::base::ModelLoadConfig, tensor::DType};

use crate::generation::GenerationOverrides;
use crate::kjarni_config::{GenerationParams, KjarniConfig};

impl KjarniConfig {
    /// Get generation overrides for a specific model and task.
    pub fn get_generation_overrides(&self, model_name: &str, task: Task) -> GenerationOverrides {
        // Start with task defaults
        let task_params = match task {
            Task::Chat => &self.chat.generation,
            Task::Generate => &self.generate.generation,
            _ => return GenerationOverrides::default(),
        };

        // Apply model-specific overrides if present
        let model_params = self
            .models
            .get(&normalize_model_name(model_name))
            .and_then(|m| m.generation.as_ref());

        // Merge: model overrides > task defaults
        merge_generation_params(task_params, model_params)
    }

    /// Get load config for a specific model.
    pub fn get_load_config(&self, model_name: &str) -> ModelLoadConfig {
        let mut config = ModelLoadConfig {
            offload_embeddings: self.load.offload_embeddings,
            offload_lm_head: self.load.offload_lm_head,
            target_dtype: parse_dtype(&self.load.dtype),
            quantize_lm_head: self
                .load
                .quantize_lm_head
                .as_ref()
                .and_then(|s| parse_quantize(s)),
            quantize_embeddings: self
                .load
                .quantize_embeddings
                .as_ref()
                .and_then(|s| parse_quantize(s)),
            use_gguf: self.load.prefer_gguf,
            max_batch_size: self.load.max_batch_size,
            max_sequence_length: self.load.max_sequence_length,
            // Download progress follows the caller's own quiet setting, which the
            // builders already thread through; this resolver has none of its own.
            quiet: false,
        };

        // Apply model-specific overrides
        if let Some(model_override) = self.models.get(&normalize_model_name(model_name)) {
            if let Some(dtype) = &model_override.dtype {
                config.target_dtype = parse_dtype(dtype);
            }
            if let Some(offload) = model_override.offload_embeddings {
                config.offload_embeddings = offload;
            }
            if let Some(offload) = model_override.offload_lm_head {
                config.offload_lm_head = offload;
            }
            if let Some(quantize) = model_override.quantize_lm_head.as_ref() {
                config.quantize_lm_head = parse_quantize(quantize);
            }
        }

        config
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Task {
    Chat,
    Generate,
    Summarize,
    Translate,
    Classify,
    Embed,
    Rerank,
    Search,
    Transcribe,
}

fn merge_generation_params(
    base: &GenerationParams,
    override_: Option<&GenerationParams>,
) -> GenerationOverrides {
    let mut result = GenerationOverrides {
        temperature: Some(base.temperature),
        max_new_tokens: Some(base.max_tokens),
        top_p: base.top_p,
        top_k: base.top_k,
        min_p: base.min_p,
        repetition_penalty: Some(base.repetition_penalty),
        ..Default::default()
    };

    if let Some(o) = override_ {
        result.temperature = Some(o.temperature);
        result.max_new_tokens = Some(o.max_tokens);
        if o.top_p.is_some() {
            result.top_p = o.top_p;
        }
        if o.top_k.is_some() {
            result.top_k = o.top_k;
        }
        if o.min_p.is_some() {
            result.min_p = o.min_p;
        }
        result.repetition_penalty = Some(o.repetition_penalty);
    }

    result
}

fn normalize_model_name(name: &str) -> String {
    // Convert "llama3.2-1b-instruct" to "llama3-2-1b-instruct" for TOML keys
    name.replace('.', "-")
}

fn parse_dtype(s: &str) -> Option<DType> {
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Some(DType::F32),
        "f16" | "float16" => Some(DType::F16),
        "bf16" | "bfloat16" => Some(DType::BF16),
        _ => None,
    }
}

fn parse_quantize(s: &str) -> Option<DType> {
    match s.to_lowercase().as_str() {
        "q8" | "q8_0" => Some(DType::Q8_0),
        // "q4" | "q4_0" => Some(DType::Q4_0),
        _ => None,
    }
}
