//! Model registry and management

use anyhow::{Result, anyhow};
use kjarni_transformers::models::{
    ModelArchitecture, ModelType, download_model_files, format_params, format_size,
    get_default_cache_dir, registry::WeightsFormat,
};
use std::path::PathBuf;

/// Information about a model for display purposes
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub model_type: ModelType,
    pub cli_name: String,
    pub architecture: ModelArchitecture,
    pub description: String,
    pub size: String,
    pub params: String,
    pub downloaded: bool,
    pub has_gguf: bool,
}

/// List all available models with their download status
pub fn list_models() -> Vec<ModelEntry> {
    let cache_dir = get_default_cache_dir();

    ModelType::all()
        .filter(|model_type| !model_type.cli_name().is_empty())
        .map(|model_type| {
            let info = model_type.info();
            ModelEntry {
                model_type,
                cli_name: model_type.cli_name().to_string(),
                architecture: info.architecture,
                description: info.description.to_string(),
                size: format_size(info.size_mb),
                params: format_params(info.params_millions),
                downloaded: model_type.is_downloaded(&cache_dir),
                has_gguf: info.paths.gguf_url.is_some(),
            }
        })
        .collect()
}

/// List models filtered by architecture
pub fn list_models_by_architecture(arch: ModelArchitecture) -> Vec<ModelEntry> {
    list_models()
        .into_iter()
        .filter(|m| m.architecture == arch)
        .collect()
}

/// Get detailed info about a specific model
pub fn get_model_info(name: &str) -> Result<ModelEntry> {
    let model_type = ModelType::from_cli_name(name).ok_or_else(|| {
        anyhow!(
            "Unknown model: '{}'. Run 'kjarni model list' to see available models.",
            name
        )
    })?;

    let cache_dir = get_default_cache_dir();
    let info = model_type.info();

    Ok(ModelEntry {
        model_type,
        cli_name: model_type.cli_name().to_string(),
        architecture: info.architecture,
        description: info.description.to_string(),
        size: format_size(info.size_mb),
        params: format_params(info.params_millions),
        downloaded: model_type.is_downloaded(&cache_dir),
        has_gguf: info.paths.gguf_url.is_some(),
    })
}

/// Download a model by CLI name
pub async fn download_model(name: &str, prefer_gguf: bool, quiet: bool) -> Result<()> {
    let model_type = ModelType::from_cli_name(name).ok_or_else(|| {
        anyhow!(
            "Unknown model: '{}'. Run 'kjarni model list' to see available models.",
            name
        )
    })?;

    let cache_dir = get_default_cache_dir();
    let model_dir = model_type.cache_dir(&cache_dir);
    let info = model_type.info();

    // Determine format
    let format = if prefer_gguf && info.paths.gguf_url.is_some() {
        WeightsFormat::GGUF
    } else {
        if prefer_gguf {
            println!("! GGUF requested but not available for this model. Downloading SafeTensors.");
        }
        WeightsFormat::SafeTensors
    };

    println!("Downloading {}...", model_type.cli_name());
    println!("  Repository: {}", model_type.repo_id());
    println!(
        "  Format:     {}",
        if matches!(format, WeightsFormat::GGUF) {
            "GGUF (Optimized)"
        } else {
            "SafeTensors (Standard)"
        }
    );
    println!("  Size:       ~{}", format_size(info.size_mb));
    println!("  Destination: {}", model_dir.display());
    println!();

    download_model_files(&model_dir, &info.paths, format, quiet).await?;

    println!();
    println!("✓ Download complete!");

    Ok(())
}

/// Check if a model is downloaded
pub fn is_model_downloaded(name: &str) -> Result<bool> {
    let model_type =
        ModelType::from_cli_name(name).ok_or_else(|| anyhow!("Unknown model: '{}'", name))?;

    let cache_dir = get_default_cache_dir();
    Ok(model_type.is_downloaded(&cache_dir))
}

/// Get the cache directory for all models
pub fn cache_dir() -> PathBuf {
    get_default_cache_dir()
}

/// Get the path where a specific model would be stored
pub fn model_path(name: &str) -> Result<PathBuf> {
    let model_type =
        ModelType::from_cli_name(name).ok_or_else(|| anyhow!("Unknown model: '{}'", name))?;

    let cache_dir = get_default_cache_dir();
    Ok(model_type.cache_dir(&cache_dir))
}

/// Validate a model name without loading it
pub fn validate_model_name(name: &str) -> bool {
    ModelType::from_cli_name(name).is_some()
}

/// Get all available model names
pub fn all_model_names() -> Vec<String> {
    ModelType::all()
        .filter(|m| !m.cli_name().is_empty())
        .map(|m| m.cli_name().to_string())
        .collect()
}

/// Get models by task group
pub fn list_models_by_group(group: &str) -> Vec<ModelEntry> {
    list_models()
        .into_iter()
        .filter(|m| m.model_type.display_group() == group)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_models_returns_entries() {
        let models = list_models();
        assert!(!models.is_empty(), "Should return at least one model");
    }
    #[test]
    fn test_list_models_all_have_cli_names() {
        let models = list_models();
        for model in &models {
            assert!(
                !model.cli_name.is_empty(),
                "All models should have CLI names"
            );
        }
    }
    #[test]
    fn test_list_models_all_have_descriptions() {
        let models = list_models();
        for model in &models {
            assert!(
                !model.description.is_empty(),
                "Model {} should have a description",
                model.cli_name
            );
        }
    }
    #[test]
    fn test_list_models_all_have_size() {
        let models = list_models();
        for model in &models {
            assert!(
                !model.size.is_empty(),
                "Model {} should have size info",
                model.cli_name
            );
            assert!(
                model.size.contains("MB") || model.size.contains("GB"),
                "Model {} size should be in MB or GB: {}",
                model.cli_name,
                model.size
            );
        }
    }
    #[test]
    fn test_list_models_all_have_params() {
        let models = list_models();
        for model in &models {
            assert!(
                !model.params.is_empty(),
                "Model {} should have params info",
                model.cli_name
            );
            assert!(
                model.params.contains('M') || model.params.contains('B'),
                "Model {} params should be in M or B: {}",
                model.cli_name,
                model.params
            );
        }
    }
    #[test]
    fn test_list_models_contains_known_models() {
        let models = list_models();
        let names: Vec<&str> = models.iter().map(|m| m.cli_name.as_str()).collect();

        assert!(
            names.contains(&"minilm-l6-v2"),
            "Should contain minilm-l6-v2"
        );
        assert!(
            names.contains(&"distilbert-sentiment"),
            "Should contain distilbert-sentiment"
        );
    }
    #[test]
    fn test_list_models_unique_cli_names() {
        let models = list_models();
        let mut names: Vec<&str> = models.iter().map(|m| m.cli_name.as_str()).collect();
        let original_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), original_len, "All CLI names should be unique");
    }
    #[test]
    fn test_list_models_by_architecture_bert() {
        let models = list_models_by_architecture(ModelArchitecture::Bert);
        assert!(!models.is_empty(), "Should have BERT models");
        for model in &models {
            assert_eq!(
                model.architecture,
                ModelArchitecture::Bert,
                "All models should be BERT architecture"
            );
        }
    }
    #[test]
    fn test_list_models_by_architecture_llama() {
        let models = list_models_by_architecture(ModelArchitecture::Llama);
        for model in &models {
            assert_eq!(
                model.architecture,
                ModelArchitecture::Llama,
                "All models should be Llama architecture"
            );
        }
    }
    #[test]
    fn test_list_models_by_architecture_t5() {
        let models = list_models_by_architecture(ModelArchitecture::T5);
        for model in &models {
            assert_eq!(
                model.architecture,
                ModelArchitecture::T5,
                "All models should be T5 architecture"
            );
        }
    }
    #[test]
    fn test_list_models_by_architecture_bart() {
        let models = list_models_by_architecture(ModelArchitecture::Bart);
        for model in &models {
            assert_eq!(
                model.architecture,
                ModelArchitecture::Bart,
                "All models should be BART architecture"
            );
        }
    }
    #[test]
    fn test_list_models_by_architecture_filters_correctly() {
        let all_models = list_models();
        let bert_models = list_models_by_architecture(ModelArchitecture::Bert);
        assert!(bert_models.len() <= all_models.len());
        let manual_count = all_models
            .iter()
            .filter(|m| m.architecture == ModelArchitecture::Bert)
            .count();
        assert_eq!(bert_models.len(), manual_count);
    }
    #[test]
    fn test_list_models_by_group_embedding() {
        let models = list_models_by_group("Embedding");
        assert!(!models.is_empty(), "Should have embedding models");
        for model in &models {
            assert_eq!(
                model.model_type.display_group(),
                "Embedding",
                "All models should be in Embedding group"
            );
        }
    }
    #[test]
    fn test_list_models_by_group_classifier() {
        let models = list_models_by_group("Classifier");
        for model in &models {
            assert_eq!(
                model.model_type.display_group(),
                "Classifier",
                "All models should be in Classifier group"
            );
        }
    }
    #[test]
    fn test_list_models_by_group_llm() {
        let models = list_models_by_group("LLM (Decoder)");
        for model in &models {
            assert_eq!(
                model.model_type.display_group(),
                "LLM (Decoder)",
                "All models should be in LLM (Decoder) group"
            );
        }
    }

    #[test]
    fn test_list_models_by_group_unknown() {
        let models = list_models_by_group("NonexistentGroup");
        assert!(models.is_empty(), "Unknown group should return empty list");
    }
    #[test]
    fn test_get_model_info_valid_model() {
        let info = get_model_info("minilm-l6-v2").unwrap();

        assert_eq!(info.cli_name, "minilm-l6-v2");
        assert_eq!(info.architecture, ModelArchitecture::Bert);
        assert!(!info.description.is_empty());
        assert!(!info.size.is_empty());
        assert!(!info.params.is_empty());
    }
    #[test]
    fn test_get_model_info_distilbert() {
        let info = get_model_info("distilbert-sentiment").unwrap();
        assert_eq!(info.cli_name, "distilbert-sentiment");
        assert_eq!(info.architecture, ModelArchitecture::Bert);
    }
    #[test]
    fn test_get_model_info_llama() {
        let info = get_model_info("llama3.2-1b-instruct").unwrap();
        assert_eq!(info.cli_name, "llama3.2-1b-instruct");
        assert_eq!(info.architecture, ModelArchitecture::Llama);
        assert!(info.has_gguf, "Llama models should have GGUF available");
    }
    #[test]
    fn test_get_model_info_unknown_model() {
        let result = get_model_info("nonexistent-model");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown model"));
        assert!(err.contains("nonexistent-model"));
    }
    #[test]
    fn test_get_model_info_empty_name() {
        let result = get_model_info("");
        assert!(result.is_err());
    }
    #[test]
    fn test_get_model_info_case_insensitive() {
        let result_lower = get_model_info("minilm-l6-v2");
        let result_mixed = get_model_info("MiniLM-L6-V2");
        let result_upper = get_model_info("MINILM-L6-V2");
        assert!(result_lower.is_ok());
        assert!(
            result_mixed.is_ok(),
            "Model lookup should be case-insensitive"
        );
        assert!(
            result_upper.is_ok(),
            "Model lookup should be case-insensitive"
        );
        assert_eq!(
            result_lower.unwrap().cli_name,
            result_mixed.unwrap().cli_name
        );
    }

    #[test]
    fn test_validate_model_name_case_insensitive() {
        assert!(validate_model_name("minilm-l6-v2"));
        assert!(validate_model_name("MiniLM-L6-V2"));
        assert!(validate_model_name("MINILM-L6-V2"));
        assert!(!validate_model_name("nonexistent-model"));
        assert!(!validate_model_name(""));
    }
    #[test]
    fn test_validate_model_name_valid() {
        assert!(validate_model_name("minilm-l6-v2"));
        assert!(validate_model_name("distilbert-sentiment"));
        assert!(validate_model_name("llama3.2-1b-instruct"));
    }
    #[test]
    fn test_validate_model_name_invalid() {
        assert!(!validate_model_name("nonexistent-model"));
        assert!(!validate_model_name(""));
        assert!(!validate_model_name("random-string-123"));
    }
    #[test]
    fn test_all_model_names_not_empty() {
        let names = all_model_names();
        assert!(!names.is_empty());
    }
    #[test]
    fn test_all_model_names_no_empty_strings() {
        let names = all_model_names();
        for name in &names {
            assert!(!name.is_empty(), "No empty model names should be returned");
        }
    }

    #[test]
    fn test_all_model_names_unique() {
        let mut names = all_model_names();
        let original_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(
            names.len(),
            original_len,
            "All model names should be unique"
        );
    }

    #[test]
    fn test_all_model_names_matches_list_models() {
        let names = all_model_names();
        let models = list_models();

        assert_eq!(names.len(), models.len());

        for name in &names {
            assert!(
                models.iter().any(|m| &m.cli_name == name),
                "Name '{}' should be in list_models",
                name
            );
        }
    }

    #[test]
    fn test_is_model_downloaded_valid_name() {
        let result = is_model_downloaded("minilm-l6-v2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_model_downloaded_invalid_name() {
        let result = is_model_downloaded("nonexistent-model");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown model"));
    }

    #[test]
    fn test_is_model_downloaded_empty_name() {
        let result = is_model_downloaded("");
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_dir_returns_path() {
        let dir = cache_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_cache_dir_contains_kjarni() {
        let dir = cache_dir();
        let path_str = dir.to_string_lossy();
        assert!(
            path_str.contains("kjarni"),
            "Cache dir should contain 'kjarni': {}",
            path_str
        );
    }

    #[test]
    fn test_cache_dir_is_absolute_or_relative_to_home() {
        let dir = cache_dir();
        assert!(
            dir.components().count() > 1,
            "Cache dir should have multiple components"
        );
    }
    #[test]
    fn test_model_path_valid_model() {
        let path = model_path("minilm-l6-v2").unwrap();

        let cache = cache_dir();
        assert!(
            path.starts_with(&cache),
            "Model path should be under cache dir"
        );

        let path_str = path.to_string_lossy();
        assert!(
            path_str.contains("sentence-transformers") || path_str.contains("MiniLM"),
            "Path should reference the model: {}",
            path_str
        );
    }

    #[test]
    fn test_model_path_invalid_model() {
        let result = model_path("nonexistent-model");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown model"));
    }

    #[test]
    fn test_model_path_different_models_different_paths() {
        let path1 = model_path("minilm-l6-v2").unwrap();
        let path2 = model_path("distilbert-sentiment").unwrap();

        assert_ne!(path1, path2, "Different models should have different paths");
    }

    #[test]
    fn test_model_path_consistent() {
        let path1 = model_path("minilm-l6-v2").unwrap();
        let path2 = model_path("minilm-l6-v2").unwrap();

        assert_eq!(path1, path2, "Same model should return consistent path");
    }

    #[test]
    fn test_model_entry_debug() {
        let info = get_model_info("minilm-l6-v2").unwrap();
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("minilm-l6-v2"));
        assert!(debug_str.contains("ModelEntry"));
    }

    #[test]
    fn test_model_entry_clone() {
        let info = get_model_info("minilm-l6-v2").unwrap();
        let cloned = info.clone();

        assert_eq!(info.cli_name, cloned.cli_name);
        assert_eq!(info.description, cloned.description);
        assert_eq!(info.size, cloned.size);
        assert_eq!(info.params, cloned.params);
    }

    #[test]
    fn test_llm_models_have_gguf() {
        let llama = get_model_info("llama3.2-1b-instruct").unwrap();
        assert!(llama.has_gguf, "Llama models should have GGUF");
    }

    #[test]
    fn test_embedding_models_no_gguf() {
        let minilm = get_model_info("minilm-l6-v2").unwrap();
        assert!(!minilm.has_gguf, "MiniLM should not have GGUF");
    }

    #[test]
    fn test_has_gguf_matches_info() {
        let models = list_models();

        for model in &models {
            let info = model.model_type.info();
            let expected_has_gguf = info.paths.gguf_url.is_some();
            assert_eq!(
                model.has_gguf, expected_has_gguf,
                "has_gguf mismatch for {}",
                model.cli_name
            );
        }
    }
    #[test]
    fn test_all_listed_models_can_get_info() {
        let models = list_models();

        for model in &models {
            let result = get_model_info(&model.cli_name);
            assert!(
                result.is_ok(),
                "Should be able to get info for listed model: {}",
                model.cli_name
            );
        }
    }

    #[test]
    fn test_all_listed_models_can_get_path() {
        let models = list_models();

        for model in &models {
            let result = model_path(&model.cli_name);
            assert!(
                result.is_ok(),
                "Should be able to get path for listed model: {}",
                model.cli_name
            );
        }
    }

    #[test]
    fn test_all_listed_models_can_check_downloaded() {
        let models = list_models();

        for model in &models {
            let result = is_model_downloaded(&model.cli_name);
            assert!(
                result.is_ok(),
                "Should be able to check download status for: {}",
                model.cli_name
            );
        }
    }

    #[test]
    fn test_model_groups_cover_all_models() {
        let all_models = list_models();
        // Must list every value `display_group()` can return. This is the third
        // place that list is written out by hand, and the second time adding a
        // model broke one of them, so any new group needs updating here, in
        // `kjarni-cli/src/commands/model.rs`, and in `display_group()` itself.
        let groups = [
            "LLM (Decoder)",
            "Seq2Seq",
            "Embedding",
            "Image Embedding",
            "Re-Ranker",
            "Classifier",
            "Generation (Decoder)",
        ];

        for model in &all_models {
            let group = model.model_type.display_group();
            assert!(
                groups.contains(&group),
                "Model {} has unknown group: {}",
                model.cli_name,
                group
            );
        }
    }
    #[test]
    fn test_model_with_dots_in_name() {
        let result = get_model_info("llama3.2-1b-instruct");
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_with_numbers() {
        let result = get_model_info("qwen2.5-0.5b-instruct");
        assert!(result.is_ok());
    }
    #[test]
    fn test_whitespace_in_model_name() {
        let result = get_model_info(" minilm-l6-v2 ");
        assert!(
            result.is_err(),
            "Should not accept whitespace in model names"
        );
    }
    #[test]
    fn test_special_characters_in_model_name() {
        let result = get_model_info("model/with/slashes");
        assert!(result.is_err());

        let result = get_model_info("model:with:colons");
        assert!(result.is_err());
    }
    #[test]
    fn test_minilm_model_details() {
        let info = get_model_info("minilm-l6-v2").unwrap();

        assert_eq!(info.architecture, ModelArchitecture::Bert);
        assert!(info.params.contains("22") || info.params.contains("M"));
        assert!(!info.has_gguf);
    }
    #[test]
    fn test_distilbart_model_details() {
        let info = get_model_info("distilbart-cnn").unwrap();

        assert_eq!(info.architecture, ModelArchitecture::Bart);
        assert!(!info.has_gguf);
    }
    #[test]
    fn test_flan_t5_model_details() {
        let info = get_model_info("flan-t5-base").unwrap();

        assert_eq!(info.architecture, ModelArchitecture::T5);
        assert!(!info.has_gguf);
    }

    #[test]
    fn test_whisper_model_details() {
        let info = get_model_info("whisper-small").unwrap();
        assert_eq!(info.architecture, ModelArchitecture::Whisper);
    }
    #[test]
    fn test_phi_model_details() {
        let info = get_model_info("phi3.5-mini").unwrap();
        assert_eq!(info.architecture, ModelArchitecture::Phi3);
        assert!(info.has_gguf, "Phi models should have GGUF");
    }
    #[test]
    fn test_qwen_model_details() {
        let info = get_model_info("qwen2.5-0.5b-instruct").unwrap();
        assert_eq!(info.architecture, ModelArchitecture::Qwen2);
        assert!(info.has_gguf, "Qwen models should have GGUF");
    }
    #[test]
    fn test_cross_encoder_model_details() {
        let info = get_model_info("minilm-l6-v2-cross-encoder").unwrap();
        assert_eq!(info.architecture, ModelArchitecture::Bert);
        assert_eq!(info.model_type.display_group(), "Re-Ranker");
    }
}
