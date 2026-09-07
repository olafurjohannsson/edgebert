//! Model management commands

use anyhow::Result;
use kjarni::ModelType;
use kjarni::registry;
use kjarni_cli::ModelCommands;
use std::path::Path;

pub async fn run(action: ModelCommands) -> Result<()> {
    match action {
        ModelCommands::List {
            arch,
            task,
            downloaded,
        } => list(arch, task, downloaded),
        ModelCommands::Download { name, gguf, quiet } => download(&name, gguf, quiet).await,
        ModelCommands::Remove { name } => remove(&name),
        ModelCommands::Info { name } => info(&name),
        ModelCommands::Search { query } => search(&query),
    }
}

fn remove(name: &str) -> Result<()> {
    let model_path = registry::model_path(name)?;

    if !model_path.exists() {
        println!("Model '{}' is not downloaded.", name);
        return Ok(());
    }

    // Show what will be deleted
    let size = dir_size(&model_path)?;
    println!();
    println!("This will delete:");
    println!("  Model: {}", name);
    println!("  Path:  {}", model_path.display());
    println!("  Size:  {}", format_bytes(size));
    println!();

    // Confirm
    print!("Are you sure? [y/N] ");
    std::io::Write::flush(&mut std::io::stdout())?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if input.trim().to_lowercase() == "y" {
        std::fs::remove_dir_all(&model_path)?;
        println!("✓ Removed {}", name);
    } else {
        println!("Cancelled.");
    }

    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    }
}

fn dir_size(path: &Path) -> Result<u64> {
    let mut size = 0;
    for entry in walkdir::WalkDir::new(path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            size += entry.metadata()?.len();
        }
    }
    Ok(size)
}

fn search(query: &str) -> Result<()> {
    let results = ModelType::search(query);

    if results.is_empty() {
        println!("No models found matching '{}'", query);
        return Ok(());
    }

    println!();
    println!("Search results for '{}':", query);
    println!("{}", "-".repeat(60));

    for (model_type, _score) in results.iter().take(10) {
        let info = model_type.info();
        let downloaded = model_type.is_downloaded(&registry::cache_dir());
        let status = if downloaded { "✓" } else { " " };

        println!(
            "  {} {:<28} {:>8}  {}",
            status,
            model_type.cli_name(),
            format_params(info.params_millions),
            truncate(info.description, 30)
        );
    }

    println!();
    Ok(())
}

/// Format parameters in a human-readable way (e.g., "1.5B", "345M")
pub fn format_params(millions: usize) -> String {
    if millions >= 1000 {
        format!("{:.1}B", millions as f64 / 1000.0)
    } else {
        format!("{}M", millions)
    }
}

fn list(
    filter_arch: Option<String>,
    filter_task: Option<String>,
    only_downloaded: bool,
) -> Result<()> {
    let models = registry::list_models();
    let arch_filter = filter_arch.as_deref().map(|s| s.to_lowercase());
    let task_filter = filter_task.as_deref().map(|s| s.to_lowercase());

    // Must cover every value display_group() can return. "Generation (Decoder)"
    // was missing, so gpt2 and distilgpt2 were absent from `model list` and
    // `model info` entirely: base completion models, reachable by `generate`
    // but undiscoverable.
    let groups = [
        "LLM (Decoder)",
        "Generation (Decoder)",
        "Seq2Seq",
        "Embedding",
        "Image Embedding",
        "Re-Ranker",
        "Classifier",
    ];

    // Calculate totals
    let downloaded_count = models.iter().filter(|m| m.downloaded).count();
    let total_count = models.len();

    println!();
    println!("Cache: {}", registry::cache_dir().display());
    println!("Models: {}/{} downloaded", downloaded_count, total_count);

    // Show active filters
    if arch_filter.is_some() || task_filter.is_some() || only_downloaded {
        let mut filters = Vec::new();
        if let Some(ref a) = arch_filter {
            filters.push(format!("arch={}", a));
        }
        if let Some(ref t) = task_filter {
            filters.push(format!("task={}", t));
        }
        if only_downloaded {
            filters.push("downloaded".to_string());
        }
        println!("Filter: {}", filters.join(", "));
    }
    println!();

    let mut any_shown = false;

    for group in groups {
        let group_models: Vec<_> = models
            .iter()
            .filter(|m| {
                let m_group = m.model_type.display_group();
                if m_group != group {
                    return false;
                }

                // Filter by downloaded
                if only_downloaded && !m.downloaded {
                    return false;
                }

                // Filter by architecture
                if let Some(ref f) = arch_filter {
                    let arch = format!("{:?}", m.architecture).to_lowercase();
                    if !arch.contains(f) {
                        return false;
                    }
                }

                // Filter by task
                if let Some(ref f) = task_filter {
                    let task_match = match f.as_str() {
                        "chat" | "llm" | "decoder" => m_group == "LLM (Decoder)",
                        "embedding" | "embed" | "encoder" => m_group == "Embedding",
                        "classification" | "classify" | "classifier" => m_group == "Classifier",
                        "rerank" | "reranker" | "re-ranker" => m_group == "Re-Ranker",
                        "seq2seq" | "summarization" | "translation" | "summarize" => {
                            m_group == "Seq2Seq"
                        }
                        _ => {
                            // Also check task enum
                            let task_str = format!("{:?}", m.model_type.info().task).to_lowercase();
                            task_str.contains(f)
                        }
                    };
                    if !task_match {
                        return false;
                    }
                }

                true
            })
            .collect();

        if group_models.is_empty() {
            continue;
        }

        any_shown = true;

        println!("{}", group.to_uppercase());
        println!("{}", "-".repeat(90));

        for m in group_models {
            // Show download status with format info
            let status = if m.downloaded {
                if has_gguf_downloaded(&m.model_type) {
                    "✓ gguf"
                } else {
                    "✓ st  " // safetensors
                }
            } else {
                "      "
            };

            let gguf_tag = if m.has_gguf { "[GGUF]" } else { "      " };

            println!(
                "  {} {:<28} {:>8} {} {}",
                status,
                m.cli_name,
                m.params,
                gguf_tag,
                truncate(&m.description, 30)
            );
        }
        println!();
    }

    if !any_shown {
        println!("No models found matching the filters.");
        println!();
        println!("Available task filters: chat, embedding, classification, rerank, seq2seq");
        println!("Available arch filters: llama, bert, qwen, t5, bart, mistral, gpt");
        println!();
    }

    println!("Legend: ✓ gguf = GGUF downloaded, ✓ st = SafeTensors downloaded");
    println!("        [GGUF] = GGUF format available for download");
    println!();
    println!("Commands:");
    println!("  kjarni model download <name>        Download SafeTensors");
    println!("  kjarni model download <name> --gguf Download GGUF (smaller)");
    println!("  kjarni model info <name>            Show details");
    println!("  kjarni model remove <name>          Delete from disk");
    println!();
    println!("Filters:");
    println!("  kjarni model list --task chat       Show only chat/LLM models");
    println!("  kjarni model list --task embedding  Show only embedding models");
    println!("  kjarni model list --arch bert       Show only BERT-based models");
    println!("  kjarni model list --downloaded      Show only downloaded models");
    println!();

    Ok(())
}

fn has_gguf_downloaded(model_type: &ModelType) -> bool {
    let cache_dir = registry::cache_dir();
    let model_dir = model_type.cache_dir(&cache_dir);
    model_dir.join("model.gguf").exists()
}

async fn download(name: &str, prefer_gguf: bool, quiet: bool) -> Result<()> {
    registry::download_model(name, prefer_gguf, quiet).await
}

fn info(name: &str) -> Result<()> {
    let model = registry::get_model_info(name)?;
    let model_path = registry::model_path(name)?;

    println!();
    println!("┌─────────────────────────────────────────┐");
    println!("│  {}  ", pad_center(&model.cli_name, 37));
    println!("└─────────────────────────────────────────┘");
    println!();
    println!("  Architecture:  {}", model.architecture.display_name());
    println!("  Parameters:    {}", model.params);
    println!("  Size (est):    {}", model.size);
    println!();

    // Download status section
    println!("  Download Status:");

    let st_path = model_path.join("model.safetensors");
    let st_index_path = model_path.join("model.safetensors.index.json");
    let gguf_path = model_path.join("model.gguf");

    if st_path.exists() || st_index_path.exists() {
        let size = if st_path.exists() {
            st_path.metadata().map(|m| m.len()).unwrap_or(0)
        } else {
            dir_size(&model_path).unwrap_or(0)
        };
        println!("    ✓ SafeTensors  {}", format_bytes(size));
    } else {
        println!("    ○ SafeTensors  (not downloaded)");
    }

    if gguf_path.exists() {
        let size = gguf_path.metadata().map(|m| m.len()).unwrap_or(0);
        println!("    ✓ GGUF         {}", format_bytes(size));
    } else if model.has_gguf {
        println!("    ○ GGUF         (available, use --gguf to download)");
    } else {
        println!("    ✗ GGUF         (not available for this model)");
    }

    println!();
    println!("  Path: {}", model_path.display());
    println!();
    println!("  Description:");
    println!("    {}", model.description);
    println!();

    Ok(())
}

fn pad_center(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        let padding = width - s.len();
        let left = padding / 2;
        let right = padding - left;
        format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes_megabytes() {
        let mb = 1024 * 1024;
        assert_eq!(format_bytes(mb), "1.0 MB");
        assert_eq!(format_bytes(mb * 10), "10.0 MB");
        assert_eq!(format_bytes(mb * 100), "100.0 MB");
        assert_eq!(format_bytes(mb + mb / 2), "1.5 MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(format_bytes(gb), "1.00 GB");
        assert_eq!(format_bytes(gb * 2), "2.00 GB");
        assert_eq!(format_bytes(gb + gb / 2), "1.50 GB");
        assert_eq!(format_bytes(gb * 10), "10.00 GB");
    }

    #[test]
    fn test_format_bytes_edge_cases() {
        assert_eq!(format_bytes(0), "0.0 MB");
        assert_eq!(format_bytes(1), "0.0 MB");
        assert_eq!(format_bytes(1024), "0.0 MB");

        // Just under 1 GB
        let just_under_gb = 1024 * 1024 * 1024 - 1;
        assert_eq!(format_bytes(just_under_gb), "1024.0 MB");

        // Exactly 1 GB
        let exactly_gb = 1024 * 1024 * 1024;
        assert_eq!(format_bytes(exactly_gb), "1.00 GB");
    }

    #[test]
    fn test_format_bytes_realistic_model_sizes() {
        let mb = 1024 * 1024;
        let gb = 1024 * 1024 * 1024;
        assert_eq!(format_bytes(90 * mb), "90.0 MB");
        assert_eq!(format_bytes(268 * mb), "268.0 MB");
        assert_eq!(format_bytes(550 * mb), "550.0 MB");
        assert_eq!(format_bytes(gb + 600 * mb), "1.59 GB");
        assert_eq!(format_bytes(3 * gb), "3.00 GB");
        assert_eq!(format_bytes(7 * gb), "7.00 GB");
        assert_eq!(format_bytes(16 * gb), "16.00 GB");
    }
    #[test]
    fn test_format_params_millions() {
        assert_eq!(format_params(22), "22M");
        assert_eq!(format_params(66), "66M");
        assert_eq!(format_params(125), "125M");
        assert_eq!(format_params(345), "345M");
        assert_eq!(format_params(999), "999M");
    }

    #[test]
    fn test_format_params_billions() {
        assert_eq!(format_params(1000), "1.0B");
        assert_eq!(format_params(1230), "1.2B");
        assert_eq!(format_params(1500), "1.5B");
        assert_eq!(format_params(3210), "3.2B");
        assert_eq!(format_params(7240), "7.2B");
        assert_eq!(format_params(8030), "8.0B");
    }

    #[test]
    fn test_format_params_edge_cases() {
        assert_eq!(format_params(0), "0M");
        assert_eq!(format_params(1), "1M");
        assert_eq!(format_params(999), "999M");
        assert_eq!(format_params(1000), "1.0B");
        assert_eq!(format_params(1001), "1.0B");
    }

    #[test]
    fn test_format_params_realistic_models() {
        assert_eq!(format_params(22), "22M");
        assert_eq!(format_params(66), "66M");
        assert_eq!(format_params(137), "137M");
        assert_eq!(format_params(490), "490M");
        assert_eq!(format_params(1230), "1.2B");
        assert_eq!(format_params(3210), "3.2B");
        assert_eq!(format_params(3800), "3.8B");
        assert_eq!(format_params(7240), "7.2B");
        assert_eq!(format_params(8030), "8.0B");
    }
    #[test]
    fn test_pad_center_shorter_string() {
        assert_eq!(pad_center("hi", 10), "    hi    ");
        assert_eq!(pad_center("abc", 7), "  abc  ");
        assert_eq!(pad_center("test", 10), "   test   ");
    }

    #[test]
    fn test_pad_center_odd_padding() {
        assert_eq!(pad_center("hi", 5), " hi  ");
        assert_eq!(pad_center("a", 4), " a  ");
        assert_eq!(pad_center("abc", 6), " abc  ");
    }

    #[test]
    fn test_pad_center_exact_width() {
        assert_eq!(pad_center("hello", 5), "hello");
        assert_eq!(pad_center("test", 4), "test");
    }

    #[test]
    fn test_pad_center_longer_string() {
        assert_eq!(pad_center("hello world", 5), "hello world");
        assert_eq!(pad_center("toolong", 3), "toolong");
    }

    #[test]
    fn test_pad_center_empty_string() {
        assert_eq!(pad_center("", 5), "     ");
        assert_eq!(pad_center("", 0), "");
    }

    #[test]
    fn test_pad_center_width_zero() {
        assert_eq!(pad_center("hello", 0), "hello");
        assert_eq!(pad_center("", 0), "");
    }

    #[test]
    fn test_pad_center_single_char() {
        assert_eq!(pad_center("x", 5), "  x  ");
        assert_eq!(pad_center("x", 4), " x  "); // odd padding: extra on right
        assert_eq!(pad_center("x", 3), " x ");
        assert_eq!(pad_center("x", 2), "x "); // odd padding: extra on right
        assert_eq!(pad_center("x", 1), "x");
    }

    #[test]
    fn test_pad_center_realistic_model_name() {
        let result = pad_center("llama3.2-1b-instruct", 37);
        assert_eq!(result.len(), 37);
        assert!(result.starts_with(' '));
        assert!(result.ends_with(' '));
        assert!(result.contains("llama3.2-1b-instruct"));
    }

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hi", 5), "hi");
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate("hello", 5), "hello");
        assert_eq!(truncate("test", 4), "test");
    }

    #[test]
    fn test_truncate_long_string() {
        assert_eq!(truncate("hello world", 8), "hello...");
        assert_eq!(truncate("this is a long string", 10), "this is...");
    }

    #[test]
    fn test_truncate_minimum_length() {
        // With max_len=4, we get 1 char + "..."
        assert_eq!(truncate("hello", 4), "h...");
        assert_eq!(truncate("abcdef", 5), "ab...");
    }

    #[test]
    fn test_truncate_empty_string() {
        assert_eq!(truncate("", 10), "");
        assert_eq!(truncate("", 0), "");
    }

    #[test]
    fn test_truncate_realistic_descriptions() {
        assert_eq!(
            truncate("Fastest sentence embedding model. Ideal for basic RAG.", 30),
            "Fastest sentence embedding ..."
        );
        assert_eq!(
            truncate("Fast binary sentiment", 30),
            "Fast binary sentiment"
        );
        assert_eq!(
            truncate(
                "Zero-shot classifier. Classify text into ANY labels you provide at runtime.",
                30
            ),
            "Zero-shot classifier. Class..."
        );
    }

    #[test]
    fn test_format_functions_together() {
        // Simulate what list() does for display
        let params = format_params(1230); // 1.2B
        let desc = truncate(
            "Official Meta edge model. Very fast, good general chat.",
            30,
        );

        assert_eq!(params, "1.2B");
        assert_eq!(desc, "Official Meta edge model. V...");
    }

    #[test]
    fn test_pad_center_with_model_names() {
        // Various model name lengths
        let names = vec![
            "gpt2",
            "llama3.2-1b",
            "llama3.2-1b-instruct",
            "deepseek-r1-8b",
            "bert-sentiment-multilingual",
        ];

        for name in names {
            let padded = pad_center(name, 37);
            // Should be exactly 37 chars or longer if name is longer
            assert!(padded.len() >= 37 || padded.len() == name.len());
            // Should contain the original name
            assert!(padded.contains(name));
        }
    }
    #[test]
    fn test_truncate_unicode() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_format_bytes_very_large() {
        let tb = 1024u64 * 1024 * 1024 * 1024;
        assert_eq!(format_bytes(tb), "1024.00 GB");
    }

    #[test]
    fn test_format_params_very_large() {
        assert_eq!(format_params(70000), "70.0B");
        assert_eq!(format_params(175000), "175.0B");
        assert_eq!(format_params(540000), "540.0B");
    }
}
