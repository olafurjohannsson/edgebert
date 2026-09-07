//! Search command with colored terminal output.

use anyhow::{Result, anyhow};
use colored::*;
use kjarni::{IndexReader, SearchMode, SearchResult, embedder::Embedder};

use crate::commands::display;

pub async fn run(
    index_path: &str,
    query: &str,
    top_k: usize,
    mode: &str,
    model: &str,
    rerank_model: Option<&str>,
    format: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // Open index
    let reader = IndexReader::open(index_path)?;

    if reader.is_empty() {
        return Err(anyhow!("Index is empty."));
    }

    if !quiet {
        eprintln!(
            "{}",
            format!(
                "Loaded index: {} documents in {} segments",
                reader.len(),
                reader.segment_count()
            )
            .dimmed()
        );
    }

    // Parse search mode
    let search_mode: SearchMode = mode.parse().map_err(|e: String| anyhow!(e))?;

    // If rerank model is provided, fetch more results initially
    let fetch_k = if rerank_model.is_some() {
        top_k * 5
    } else {
        top_k
    };

    // Search based on mode
    let mut results = match search_mode {
        SearchMode::Keyword => {
            if !quiet {
                eprintln!("{}", "Searching with BM25...".dimmed());
            }
            reader.search_keywords(query, fetch_k)
        }
        SearchMode::Semantic => {
            let query_embedding = get_query_embedding(query, model, &reader, gpu, quiet).await?;
            if !quiet {
                eprintln!("{}", "Searching semantically...".dimmed());
            }
            reader.search_semantic(&query_embedding, fetch_k)
        }
        SearchMode::Hybrid => {
            let query_embedding = get_query_embedding(query, model, &reader, gpu, quiet).await?;
            if !quiet {
                eprintln!("{}", "Searching with hybrid (BM25 + semantic)...".dimmed());
            }
            reader.search_hybrid(query, &query_embedding, fetch_k)
        }
    };

    if results.is_empty() {
        if !quiet {
            eprintln!("{}", "No results found.".dimmed());
        }
        return Ok(());
    }

    // Optional rerank
    if let Some(reranker_name) = rerank_model {
        if !quiet {
            eprintln!(
                "{}",
                format!(
                    "Reranking top {} results with '{}'...",
                    results.len(),
                    reranker_name
                )
                .dimmed()
            );
        }

        let mut builder = kjarni::reranker::Reranker::builder(reranker_name).quiet(quiet);
        if gpu {
            builder = builder.gpu();
        }

        let reranker = builder.build().await.map_err(|e| anyhow!(e))?;
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        let reranked_results = reranker
            .rerank(query, &texts)
            .await
            .map_err(|e| anyhow!(e))?;

        let mut new_results = Vec::with_capacity(reranked_results.len());
        for rr in reranked_results {
            let mut original_result = results[rr.index].clone();
            original_result.score = rr.score;
            new_results.push(original_result);
        }

        if new_results.len() > top_k {
            new_results.truncate(top_k);
        }

        results = new_results;
    }

    // Output results
    let output = format_results(&results, format, query)?;
    print!("{}", output);

    Ok(())
}

async fn get_query_embedding(
    query: &str,
    model: &str,
    reader: &IndexReader,
    gpu: bool,
    quiet: bool,
) -> Result<Vec<f32>> {
    let mut builder = Embedder::builder(model).quiet(quiet);

    if gpu {
        builder = builder.gpu();
    } else {
        builder = builder.cpu();
    }

    let embedder = builder.build().await.map_err(|e| anyhow!(e))?;

    if embedder.dimension() != reader.dimension() {
        return Err(anyhow!(
            "Dimension mismatch: index expects {}, model '{}' produces {}.\n\
             Use the same model that created the index.",
            reader.dimension(),
            embedder.model_name(),
            embedder.dimension()
        ));
    }

    let embedding = embedder.embed(query).await.map_err(|e| anyhow!(e))?;
    Ok(embedding)
}

fn format_results(results: &[SearchResult], format: &str, query: &str) -> Result<String> {
    match format {
        "json" => format_results_json(results),
        "jsonl" => format_results_jsonl(results),
        "text" => Ok(format_results_pretty(results, query)),
        "docs" => Ok(format_results_docs(results)),
        _ => Err(anyhow!(
            "Unknown format: '{}'. Use: json, jsonl, text, docs",
            format
        )),
    }
}

fn format_results_pretty(results: &[SearchResult], query: &str) -> String {
    let mut output = String::new();

    output.push_str(&format!(
        "\n  {} \"{}\"\n\n",
        "Results for".dimmed(),
        query.white().bold()
    ));

    // Bars are drawn relative to the best hit, not stretched across the range of
    // this page. Min-max normalising forced the last result to exactly 0.0%
    // however relevant it was, which reads as "no match" rather than "ranked
    // last". The three score types reaching here are all positive: reranker
    // probabilities, cosine similarity, and reciprocal-rank-fusion weights, whose
    // absolute values differ by an order of magnitude and are not comparable
    // across modes, which is why this is a bar and not a percentage claim.
    let max_score = results
        .iter()
        .map(|r| r.score)
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-6);

    for (i, r) in results.iter().enumerate() {
        let norm_score = (r.score / max_score).clamp(0.0, 1.0);

        let source = r
            .metadata
            .get("source")
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        // Rank + title
        output.push_str(&format!(
            "  {} {}\n",
            display::rank_label(i + 1),
            source.white().bold()
        ));

        // Bar + percentage
        output.push_str(&format!(
            "       {}  {}\n",
            display::score_bar(norm_score, 20),
            display::score_pct(norm_score)
        ));

        // Snippet
        let text_snippet = display::snippet(&r.text, 72);
        output.push_str(&format!("       \"{}\"\n", text_snippet));

        output.push('\n');
    }

    output
}

fn format_results_json(results: &[SearchResult]) -> Result<String> {
    let output: Vec<_> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "score": r.score,
                "document_id": r.document_id,
                "text": r.text,
                "metadata": r.metadata
            })
        })
        .collect();
    Ok(format!("{}\n", serde_json::to_string_pretty(&output)?))
}

fn format_results_jsonl(results: &[SearchResult]) -> Result<String> {
    let mut output = String::new();
    for r in results {
        let obj = serde_json::json!({
            "score": r.score,
            "document_id": r.document_id,
            "text": r.text,
            "metadata": r.metadata
        });
        output.push_str(&serde_json::to_string(&obj)?);
        output.push('\n');
    }
    Ok(output)
}

fn format_results_docs(results: &[SearchResult]) -> String {
    let mut output = String::new();
    for r in results {
        output.push_str(&r.text);
        output.push('\n');
    }
    output
}
