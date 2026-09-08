//! Searching a photo directory by description.
//!
//! CLIP embeds images and text into one space, so a sentence and a picture that
//! matches it land near each other. Index once, then query by typing.

use anyhow::{Context, Result};
use colored::Colorize;
use kjarni::ImageIndex;
use std::path::{Path, PathBuf};

use crate::commands::display;

/// Where the CLIP checkpoint lives once `kjarni model download` has run.
fn model_dir() -> Result<PathBuf> {
    let dir = kjarni::registry::cache_dir().join("laion_CLIP-ViT-B-32-laion2B-s34B-b79K");
    anyhow::ensure!(
        dir.is_dir(),
        "clip-vit-base-32 is not downloaded.\n  run: kjarni model download clip-vit-base-32"
    );
    Ok(dir)
}

/// Walks `inputs` and writes one vector per image to `output`.
pub async fn index(inputs: &[String], output: &str, quiet: bool) -> Result<()> {
    let dir = model_dir()?;
    if !quiet {
        println!("{}", "Loading CLIP...".dimmed());
    }
    let mut index = ImageIndex::load(&dir).context("loading CLIP")?;

    let mut total = 0usize;
    let mut failures = Vec::new();
    for input in inputs {
        let path = Path::new(input);
        let report = if path.is_dir() {
            index.add_directory(path)?
        } else {
            // A single file still goes through add_image, so an unreadable one
            // reports the same way a directory scan would.
            match index.add_image(path) {
                Ok(()) => kjarni::ScanReport {
                    added: 1,
                    ..Default::default()
                },
                Err(e) => kjarni::ScanReport {
                    skipped: 1,
                    failures: vec![(path.to_path_buf(), format!("{e:#}"))],
                    ..Default::default()
                },
            }
        };
        total += report.added;
        failures.extend(report.failures);
        if !quiet {
            println!("  {input}: {} indexed", report.added);
        }
    }

    // Named individually rather than counted: a photo library with three
    // unreadable files should say which three.
    if !failures.is_empty() {
        eprintln!(
            "\n{} {} could not be read:",
            "warning:".yellow(),
            failures.len()
        );
        for (path, why) in &failures {
            eprintln!("  {}: {why}", path.display());
        }
    }

    index
        .save(Path::new(output))
        .with_context(|| format!("writing {output}"))?;

    if !quiet {
        println!("\n{} {total} images -> {output}", "Indexed".green().bold());
    }
    Ok(())
}

/// Ranks an existing index against a description.
pub async fn search(index_path: &str, query: &str, top_k: usize, quiet: bool) -> Result<()> {
    let dir = model_dir()?;
    let mut index = ImageIndex::load(&dir).context("loading CLIP")?;
    let n = index
        .load_entries(Path::new(index_path))
        .with_context(|| format!("reading {index_path}"))?;

    if !quiet {
        println!("{}", format!("Searching {n} images...").dimmed());
    }

    let hits = index.search(query, top_k)?;
    if hits.is_empty() {
        println!("No images indexed.");
        return Ok(());
    }

    println!("\n  {} {}\n", "Results for".dimmed(), query.white().bold());

    // Relative to the best hit. CLIP similarities are small in absolute terms
    // even for a good match, typically 0.2 to 0.35, so the number is only
    // meaningful as a ranking and the bar says so.
    let best = hits
        .iter()
        .map(|h| h.score)
        .fold(f32::MIN, f32::max)
        .max(1e-6);
    for (i, hit) in hits.iter().enumerate() {
        println!("    {}. {}", i + 1, hit.path.display().to_string().cyan());
        println!(
            "       {}  {:.4}",
            display::score_bar((hit.score / best).clamp(0.0, 1.0), 20),
            hit.score
        );
    }
    println!();
    Ok(())
}
