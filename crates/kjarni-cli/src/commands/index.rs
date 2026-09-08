//! Index management commands
use anyhow::{Result, anyhow};
use kjarni::Chunk;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::sync::mpsc;

use kjarni::embedder::Embedder;
use kjarni::{DocumentLoader, IndexConfig, IndexReader, IndexWriter, LoaderConfig, SplitterConfig};
use kjarni_cli::IndexCommands;

const ENCODE_BATCH_SIZE: usize = 32;

pub async fn run(action: IndexCommands) -> Result<()> {
    match action {
        IndexCommands::Create {
            output,
            from_chunks,
            inputs,
            chunk_size,
            chunk_overlap,
            model,
            gpu,
            quiet,
        } => {
            create(
                &output,
                from_chunks,
                &inputs,
                chunk_size,
                chunk_overlap,
                &model,
                gpu,
                quiet,
            )
            .await
        }
        IndexCommands::Add {
            index_path,
            inputs,
            chunk_size,
            chunk_overlap,
            model,
            gpu,
            quiet,
        } => {
            add(
                &index_path,
                &inputs,
                chunk_size,
                chunk_overlap,
                &model,
                gpu,
                quiet,
            )
            .await
        }
        IndexCommands::Info { index_path } => info(&index_path),
    }
}

/// Create a new index from documents
async fn create(
    output: &str,
    from_chunks: Option<String>,
    inputs: &[String],
    chunk_size: usize,
    chunk_overlap: usize,
    model: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    if inputs.is_empty() && from_chunks.is_none() {
        return Err(anyhow!(
            "No input files specified. Provide files or directories to index, \
             or --from-chunks with a JSONL file."
        ));
    }

    // Load embedder (Handles download, validation, device automatically)
    let embedder = load_embedder(model, gpu, quiet).await?;
    let dimension = embedder.dimension();

    if !quiet {
        eprintln!("Model: {}", embedder.model_name());
        eprintln!("Dimension: {}", dimension);
    }

    // Create index writer
    let config = IndexConfig {
        dimension,
        max_docs_per_segment: 10_000,
        embedding_model: Some(model.to_string()),
        ..Default::default()
    };

    let mut writer = IndexWriter::open(output, config)?;

    // Process documents
    let total_indexed = match &from_chunks {
        Some(path) => process_chunk_file(&mut writer, &embedder, path, quiet).await?,
        None => {
            process_inputs(
                &mut writer,
                &embedder,
                inputs,
                chunk_size,
                chunk_overlap,
                quiet,
            )
            .await?
        }
    };

    if total_indexed == 0 {
        return Err(anyhow!("No documents found to index."));
    }

    // Commit
    writer.commit()?;

    if !quiet {
        eprintln!();
        eprintln!("✓ Index created: {}", output);
        eprintln!("  Documents: {}", total_indexed);
        eprintln!("  Dimension: {}", dimension);

        if let Ok(size) = calculate_index_size(output) {
            eprintln!("  Size: {}", format_size(size));
        }
    }

    Ok(())
}

/// Add documents to an existing index
async fn add(
    index_path: &str,
    inputs: &[String],
    chunk_size: usize,
    chunk_overlap: usize,
    model: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    if inputs.is_empty() {
        return Err(anyhow!("No input files specified."));
    }

    // Open existing index
    let mut writer = IndexWriter::open_existing(index_path)?;
    let initial_count = writer.len();
    let dimension = writer.dimension();

    if !quiet {
        eprintln!("Opened index with {} documents", initial_count);
    }

    // Load embedder
    let embedder = load_embedder(model, gpu, quiet).await?;

    // Verify dimension matches
    if embedder.dimension() != dimension {
        return Err(anyhow!(
            "Dimension mismatch: index has {}, model '{}' produces {}.\n\
             Use the same model that was used to create the index.",
            dimension,
            embedder.model_name(),
            embedder.dimension()
        ));
    }

    // Process new documents
    let added = process_inputs(
        &mut writer,
        &embedder,
        inputs,
        chunk_size,
        chunk_overlap,
        quiet,
    )
    .await?;

    // Commit
    writer.commit()?;

    if !quiet {
        eprintln!();
        eprintln!("✓ Added {} documents to index", added);
        eprintln!("  Total documents: {}", initial_count + added);
    }

    Ok(())
}

/// Indexes a JSONL file of already-chunked documents, one object per line.
///
/// `{"text": "...", "metadata": {"k": "v"}}`, where `metadata` is optional and its
/// values are strings. Chunking is skipped entirely, which is the point: a caller
/// who has already split documents their own way should not have them re-split.
///
/// The flag existed and was accepted for some time while the handler ignored it,
/// so `--from-chunks` parsed and then failed with "No input files specified".
async fn process_chunk_file(
    writer: &mut IndexWriter,
    embedder: &Embedder,
    path: &str,
    quiet: bool,
) -> Result<usize> {
    use std::io::BufRead;

    let file =
        std::fs::File::open(path).map_err(|e| anyhow!("could not open chunk file {path}: {e}"))?;
    let reader = std::io::BufReader::new(file);

    let mut texts: Vec<String> = Vec::new();
    let mut metadata: Vec<HashMap<String, String>> = Vec::new();
    let mut total = 0usize;

    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| anyhow!("reading {path} line {}: {e}", lineno + 1))?;
        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| anyhow!("{path} line {}: {e}", lineno + 1))?;

        // A line without text is a mistake worth naming rather than skipping: it
        // would otherwise show up later as an index that is quietly short.
        let text = value
            .get("text")
            .and_then(|t| t.as_str())
            .ok_or_else(|| anyhow!("{path} line {}: no \"text\" field", lineno + 1))?;

        let mut meta = HashMap::new();
        if let Some(obj) = value.get("metadata").and_then(|m| m.as_object()) {
            for (k, v) in obj {
                if let Some(s) = v.as_str() {
                    meta.insert(k.clone(), s.to_string());
                }
            }
        }

        texts.push(text.to_string());
        metadata.push(meta);

        if texts.len() >= ENCODE_BATCH_SIZE {
            total += flush_batch(writer, embedder, &mut texts, &mut metadata).await?;
            if !quiet && std::io::IsTerminal::is_terminal(&std::io::stderr()) {
                eprint!("\r  Indexed {total} documents");
            }
        }
    }

    if !texts.is_empty() {
        total += flush_batch(writer, embedder, &mut texts, &mut metadata).await?;
    }

    if !quiet && total > 0 {
        if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            eprintln!("\r  Indexed {total} documents");
        } else {
            eprintln!("  Indexed {total} documents");
        }
    }
    Ok(total)
}

async fn process_inputs(
    writer: &mut IndexWriter,
    embedder: &Embedder,
    inputs: &[String],
    chunk_size: usize,
    chunk_overlap: usize,
    quiet: bool,
) -> Result<usize> {
    let loader_config = LoaderConfig {
        splitter: SplitterConfig {
            chunk_size,
            chunk_overlap,
            clean_markdown: true,
        },
        quiet,
        ..Default::default()
    };
    let loader = Arc::new(DocumentLoader::new(loader_config));

    let (tx, mut rx) = mpsc::channel::<Vec<Chunk>>(64);
    let concurrency_limit = 16;
    let semaphore = Arc::new(Semaphore::new(concurrency_limit));

    // Producer Task
    let inputs_owned = inputs.to_vec();
    let loader_ref = loader.clone();

    tokio::spawn(async move {
        for input in inputs_owned {
            let path = Path::new(&input);
            if !path.exists() {
                continue;
            }

            for entry in walkdir::WalkDir::new(path)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if !entry.file_type().is_file() {
                    continue;
                }

                if !loader_ref.is_supported_extension(entry.path()) {
                    continue;
                }

                let permit = match semaphore.clone().acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => break,
                };

                let loader = loader_ref.clone();
                let tx = tx.clone();
                let file_path = entry.path().to_owned();

                tokio::spawn(async move {
                    let _p = permit;

                    let result =
                        tokio::task::spawn_blocking(move || loader.load_file(&file_path)).await;

                    if let Ok(Ok(chunks)) = result
                        && !chunks.is_empty()
                    {
                        let _ = tx.send(chunks).await;
                    }
                });
            }
        }
    });

    // Consumer Loop
    let mut batch_texts: Vec<String> = Vec::with_capacity(ENCODE_BATCH_SIZE);
    let mut batch_metadata: Vec<HashMap<String, String>> = Vec::with_capacity(ENCODE_BATCH_SIZE);
    let mut total_indexed = 0;

    while let Some(chunks) = rx.recv().await {
        for chunk in chunks {
            batch_texts.push(chunk.text);
            batch_metadata.push(chunk.metadata.to_hashmap());

            if batch_texts.len() >= ENCODE_BATCH_SIZE {
                total_indexed +=
                    flush_batch(writer, embedder, &mut batch_texts, &mut batch_metadata).await?;

                // Carriage-return progress only makes sense on a terminal. Piped
                // or redirected it emits one line per batch with no newline
                // between them, which for the Python docs was a thousand copies
                // of "Indexed N documents" run together in the log.
                if !quiet && std::io::IsTerminal::is_terminal(&std::io::stderr()) {
                    eprint!("\r  Indexed {} documents", total_indexed);
                }
            }
        }
    }

    if !batch_texts.is_empty() {
        total_indexed +=
            flush_batch(writer, embedder, &mut batch_texts, &mut batch_metadata).await?;
    }

    // The final count is worth having in a log, so this one is unconditional; only
    // the leading \r is dropped when nothing will consume it.
    if !quiet && total_indexed > 0 {
        if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            eprintln!("\r  Indexed {} documents", total_indexed);
        } else {
            eprintln!("  Indexed {} documents", total_indexed);
        }
    }

    Ok(total_indexed)
}

/// Encode a batch and write to index
async fn flush_batch(
    writer: &mut IndexWriter,
    embedder: &Embedder,
    texts: &mut Vec<String>,
    metadata: &mut Vec<HashMap<String, String>>,
) -> Result<usize> {
    let count = texts.len();

    let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed_batch(&texts_ref).await?;

    for ((text, meta), embedding) in texts.drain(..).zip(metadata.drain(..)).zip(embeddings) {
        writer.add(&text, &embedding, Some(&meta))?;
    }

    Ok(count)
}

/// Show index information
fn info(index_path: &str) -> Result<()> {
    let reader = IndexReader::open(index_path)?;
    let output = format_index_info(index_path, &reader)?;
    print!("{}", output);
    Ok(())
}

/// Format index information as a string
fn format_index_info(index_path: &str, reader: &IndexReader) -> Result<String> {
    let mut output = String::new();

    output.push('\n');
    output.push_str(&format!("Index: {}\n", index_path));
    output.push_str(&format!("{}\n", "-".repeat(50)));
    output.push_str(&format!("Documents:      {}\n", reader.len()));
    output.push_str(&format!("Segments:       {}\n", reader.segment_count()));
    output.push_str(&format!("Dimension:      {}\n", reader.dimension()));

    if let Ok(size) = calculate_index_size(index_path) {
        output.push_str(&format!("Total size:     {}\n", format_size(size)));
    }

    if !reader.is_empty() {
        output.push('\n');
        output.push_str("Sample documents:\n");
        for i in 0..3.min(reader.len()) {
            if let Ok(doc) = reader.get_document(i) {
                output.push_str(&format!("  [{}] {}\n", i, truncate(&doc, 60)));
            }
        }
    }

    // Show segment details
    output.push('\n');
    output.push_str("Segments:\n");
    let segments_dir = Path::new(index_path).join("segments");
    if segments_dir.exists() {
        let mut entries: Vec<_> = std::fs::read_dir(&segments_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let meta_path = entry.path().join("segment.json");
            if let Ok(content) = std::fs::read_to_string(&meta_path)
                && let Ok(meta) = serde_json::from_str::<serde_json::Value>(&content)
            {
                let doc_count = meta["doc_count"].as_u64().unwrap_or(0);
                let name = entry.file_name();
                output.push_str(&format!("  {:?}: {} docs\n", name, doc_count));
            }
        }
    }

    output.push('\n');
    Ok(output)
}

async fn load_embedder(model: &str, gpu: bool, quiet: bool) -> Result<Embedder> {
    let mut builder = Embedder::builder(model).quiet(quiet);

    if gpu {
        builder = builder.gpu();
    } else {
        builder = builder.cpu();
    }

    builder.build().await.map_err(|e| anyhow!(e))
}

fn calculate_index_size(path: &str) -> Result<u64> {
    let mut total = 0u64;
    for entry in walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            total += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    Ok(total)
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} bytes", bytes)
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace(['\n', '\t'], " ");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 bytes");
        assert_eq!(format_size(1), "1 bytes");
        assert_eq!(format_size(512), "512 bytes");
        assert_eq!(format_size(999), "999 bytes");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1_000), "1.00 KB");
        assert_eq!(format_size(1_500), "1.50 KB");
        assert_eq!(format_size(10_000), "10.00 KB");
        assert_eq!(format_size(999_999), "1000.00 KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1_000_000), "1.00 MB");
        assert_eq!(format_size(1_500_000), "1.50 MB");
        assert_eq!(format_size(100_000_000), "100.00 MB");
        assert_eq!(format_size(999_999_999), "1000.00 MB");
    }

    #[test]
    fn test_format_size_gigabytes() {
        assert_eq!(format_size(1_000_000_000), "1.00 GB");
        assert_eq!(format_size(1_500_000_000), "1.50 GB");
        assert_eq!(format_size(10_000_000_000), "10.00 GB");
        assert_eq!(format_size(100_000_000_000), "100.00 GB");
    }

    #[test]
    fn test_format_size_edge_cases() {
        // Just under KB threshold
        assert_eq!(format_size(999), "999 bytes");
        // Just at KB threshold
        assert_eq!(format_size(1_000), "1.00 KB");
        // Just under MB threshold
        assert_eq!(format_size(999_999), "1000.00 KB");
        // Just at MB threshold
        assert_eq!(format_size(1_000_000), "1.00 MB");
    }

    #[test]
    fn test_format_size_precision() {
        assert_eq!(format_size(1_234_567), "1.23 MB");
        assert_eq!(format_size(1_235_000), "1.24 MB"); // Rounding
        assert_eq!(format_size(1_234_567_890), "1.23 GB");
    }

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate("hello", 5), "hello");
        assert_eq!(truncate("hello world", 11), "hello world");
    }

    #[test]
    fn test_truncate_long_string() {
        assert_eq!(truncate("hello world", 8), "hello...");
        assert_eq!(truncate("this is a very long string", 10), "this is...");
    }

    #[test]
    fn test_truncate_replaces_newlines() {
        assert_eq!(truncate("hello\nworld", 20), "hello world");
        assert_eq!(truncate("line1\nline2\nline3", 20), "line1 line2 line3");
    }

    #[test]
    fn test_truncate_replaces_tabs() {
        assert_eq!(truncate("hello\tworld", 20), "hello world");
        assert_eq!(truncate("col1\tcol2\tcol3", 20), "col1 col2 col3");
    }

    #[test]
    fn test_truncate_mixed_whitespace() {
        assert_eq!(truncate("hello\n\tworld", 20), "hello  world");
        assert_eq!(truncate("a\nb\tc\nd", 10), "a b c d");
    }

    #[test]
    fn test_truncate_with_newlines_then_truncate() {
        let input = "line one\nline two\nline three";
        let result = truncate(input, 15);
        assert_eq!(result, "line one lin...");
    }

    #[test]
    fn test_truncate_empty_string() {
        assert_eq!(truncate("", 10), "");
        assert_eq!(truncate("", 0), "");
    }

    #[test]
    fn test_truncate_unicode() {
        assert_eq!(truncate("héllo", 10), "héllo");
        assert_eq!(truncate("日本語テスト", 20), "日本語テスト");
    }

    #[test]
    fn test_truncate_only_whitespace() {
        assert_eq!(truncate("\n\n\n", 10), "   ");
        assert_eq!(truncate("\t\t\t", 10), "   ");
    }

    #[test]
    fn test_truncate_minimum_length() {
        assert_eq!(truncate("hello", 4), "h...");
    }

    #[test]
    fn test_truncate_realistic_document() {
        let doc = "This is a document with multiple paragraphs.\n\nIt has some content that spans several lines.\n\nAnd more content here.";
        let result = truncate(doc, 60);

        assert_eq!(result.len(), 60);
        assert!(result.ends_with("..."));
        assert!(!result.contains('\n'));
    }

    #[test]
    fn test_format_size_realistic_file_sizes() {
        // Common file sizes
        assert_eq!(format_size(4_096), "4.10 KB"); // 4KB block
        assert_eq!(format_size(65_536), "65.54 KB"); // 64KB
        assert_eq!(format_size(1_048_576), "1.05 MB"); // 1 MiB
        assert_eq!(format_size(104_857_600), "104.86 MB"); // 100 MiB
        assert_eq!(format_size(1_073_741_824), "1.07 GB"); // 1 GiB
    }
    #[test]
    fn test_encode_batch_size_is_reasonable() {
        const { assert!(ENCODE_BATCH_SIZE > 0) };
        const { assert!(ENCODE_BATCH_SIZE <= 128) };
        assert_eq!(ENCODE_BATCH_SIZE, 32);
    }

    /// The JSONL shape `--from-chunks` accepts. Parsing is asserted separately
    /// from indexing so a malformed line can be checked without loading a model.
    fn parse_line(line: &str) -> Result<(String, HashMap<String, String>)> {
        let value: serde_json::Value = serde_json::from_str(line)?;
        let text = value
            .get("text")
            .and_then(|t| t.as_str())
            .ok_or_else(|| anyhow!("no \"text\" field"))?;
        let mut meta = HashMap::new();
        if let Some(obj) = value.get("metadata").and_then(|m| m.as_object()) {
            for (k, v) in obj {
                if let Some(sv) = v.as_str() {
                    meta.insert(k.clone(), sv.to_string());
                }
            }
        }
        Ok((text.to_string(), meta))
    }

    #[test]
    fn a_chunk_line_needs_only_text() {
        let (text, meta) = parse_line(r#"{"text": "hello"}"#).unwrap();
        assert_eq!(text, "hello");
        assert!(meta.is_empty(), "metadata is optional");
    }

    #[test]
    fn string_metadata_is_carried_through() {
        let (_, meta) =
            parse_line(r#"{"text": "x", "metadata": {"source": "a.txt", "page": "3"}}"#).unwrap();
        assert_eq!(meta.get("source").map(String::as_str), Some("a.txt"));
        assert_eq!(meta.get("page").map(String::as_str), Some("3"));
    }

    /// Non-string metadata values are dropped rather than stringified, because the
    /// index stores strings and guessing a format for a number or an object would
    /// be inventing data.
    #[test]
    fn non_string_metadata_is_dropped_not_coerced() {
        let (_, meta) =
            parse_line(r#"{"text": "x", "metadata": {"n": 3, "ok": "yes", "sub": {"a": 1}}}"#)
                .unwrap();
        assert_eq!(meta.len(), 1);
        assert_eq!(meta.get("ok").map(String::as_str), Some("yes"));
    }

    /// A line with no text would otherwise become an index that is quietly short.
    #[test]
    fn a_line_without_text_is_an_error() {
        let err = parse_line(r#"{"body": "oops"}"#).unwrap_err().to_string();
        assert!(err.contains("text"), "{err}");
    }

    #[test]
    fn malformed_json_is_an_error() {
        assert!(parse_line("{not json").is_err());
        assert!(parse_line("").is_err());
    }
}
