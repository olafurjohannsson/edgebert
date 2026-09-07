//! Searching a photo directory by description.
//!
//! CLIP trains an image tower and a text tower together so that a picture and a
//! sentence describing it land near each other. That is the whole trick this
//! builds on: embed every photo once, embed the query when it is typed, and rank
//! by cosine. No captions, no tagging, no OCR.
//!
//! Behind `image-io`, because walking a directory means decoding files.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use kjarni_rag::{DocumentLoader, KIND_IMAGE, KIND_KEY, LoaderConfig};
use serde::{Deserialize, Serialize};

use super::config::ClipConfig;
use super::model::ClipVisionModel;
use super::text::ClipTextModel;
use super::tokenizer::ClipTokenizer;

/// One indexed image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedImage {
    pub path: PathBuf,
    /// Unit-length, so a dot product is the cosine.
    pub embedding: Vec<f32>,
    /// Words recovered from the filename, kept for keyword matching alongside the
    /// vector. Often the only text a photo has.
    pub filename_text: String,
}

/// A ranked result.
#[derive(Debug, Clone)]
pub struct ImageHit {
    pub path: PathBuf,
    pub score: f32,
}

/// What a scan did, including the files it could not read.
#[derive(Debug, Default)]
pub struct ScanReport {
    pub added: usize,
    pub skipped: usize,
    /// Path and reason, so a caller can show which photos were dropped rather
    /// than silently indexing fewer than the user has.
    pub failures: Vec<(PathBuf, String)>,
}

pub struct ImageIndex {
    vision: ClipVisionModel,
    text: ClipTextModel,
    tokenizer: ClipTokenizer,
    entries: Vec<IndexedImage>,
    /// Path to position, so rescanning a directory updates rather than duplicates.
    by_path: HashMap<PathBuf, usize>,
}

impl ImageIndex {
    /// Loads both towers and the tokenizer from a model directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = ClipConfig::from_json(
            &std::fs::read_to_string(model_dir.join("config.json"))
                .with_context(|| format!("reading config.json in {}", model_dir.display()))?,
        )?;
        let weights = kjarni_transformers::weights::ModelWeights::new(model_dir)?;

        Ok(Self {
            vision: ClipVisionModel::from_dir(model_dir)?,
            text: ClipTextModel::from_weights(&config, &weights)?,
            tokenizer: ClipTokenizer::from_json_str(
                &std::fs::read_to_string(model_dir.join("tokenizer.json"))
                    .context("reading tokenizer.json")?,
            )?,
            entries: Vec::new(),
            by_path: HashMap::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[IndexedImage] {
        &self.entries
    }

    /// Embeds one image and adds or replaces it.
    pub fn add_image(&mut self, path: &Path) -> Result<()> {
        let embedding = self
            .vision
            .embed_image_file(path)
            .with_context(|| format!("embedding {}", path.display()))?;
        let entry = IndexedImage {
            path: path.to_path_buf(),
            embedding: embedding.to_vec(),
            filename_text: filename_words(path),
        };
        match self.by_path.get(path) {
            Some(&at) => self.entries[at] = entry,
            None => {
                self.by_path.insert(path.to_path_buf(), self.entries.len());
                self.entries.push(entry);
            }
        }
        Ok(())
    }

    /// Walks a directory and indexes every image in it.
    ///
    /// A file that cannot be decoded is recorded and skipped, not fatal: one
    /// truncated download should not abandon a scan of ten thousand photos.
    pub fn add_directory(&mut self, dir: &Path) -> Result<ScanReport> {
        let loader = DocumentLoader::new(LoaderConfig {
            quiet: true,
            ..LoaderConfig::default().with_images()
        });
        let chunks = loader
            .load_directory(dir)
            .with_context(|| format!("scanning {}", dir.display()))?;

        let mut report = ScanReport::default();
        for chunk in chunks {
            if chunk.metadata.custom.get(KIND_KEY).map(String::as_str) != Some(KIND_IMAGE) {
                continue;
            }
            let Some(source) = chunk.metadata.source.as_ref() else {
                continue;
            };
            let path = PathBuf::from(source);
            match self.add_image(&path) {
                Ok(()) => report.added += 1,
                Err(e) => {
                    report.skipped += 1;
                    report.failures.push((path, format!("{e:#}")));
                }
            }
        }
        Ok(report)
    }

    /// Embeds a query the way CLIP expects, ready to compare against images.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let ids = self.tokenizer.encode(query);
        anyhow::ensure!(ids.len() >= 2, "query produced no tokens");
        // The end-of-text position is what CLIP pools: under a causal mask it is
        // the only one that has seen the whole query.
        Ok(self.text.embed_ids(&ids, ids.len() - 1)?.to_vec())
    }

    /// Ranks indexed images against a description.
    ///
    /// Scores are cosine similarities and they are small in absolute terms even
    /// for a good match, typically 0.2 to 0.35. Only the ordering is meaningful;
    /// treating the raw number as a confidence will mislead.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<ImageHit>> {
        let q = self.embed_query(query)?;
        let mut hits: Vec<ImageHit> = self
            .entries
            .iter()
            .map(|e| ImageHit {
                path: e.path.clone(),
                // Both sides are unit length, so the dot product is the cosine.
                score: e.embedding.iter().zip(&q).map(|(a, b)| a * b).sum(),
            })
            .collect();

        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Writes the index beside whatever the caller likes.
    ///
    /// Vectors only. The images stay where they are, so this is a few kilobytes
    /// per thousand photos rather than a copy of the library.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_vec(&self.entries).context("serialising index")?;
        std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))
    }

    /// Restores previously computed vectors, leaving the models as loaded.
    pub fn load_entries(&mut self, path: &Path) -> Result<usize> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading index {}", path.display()))?;
        let entries: Vec<IndexedImage> = serde_json::from_slice(&bytes).context("parsing index")?;

        let expected = self.vision.config().projection_dim;
        if let Some(bad) = entries.iter().find(|e| e.embedding.len() != expected) {
            anyhow::bail!(
                "index holds {}-dimensional vectors but this model produces {expected}; \
                 it was probably built with a different CLIP checkpoint ({})",
                bad.embedding.len(),
                bad.path.display()
            );
        }

        self.by_path = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.path.clone(), i))
            .collect();
        self.entries = entries;
        Ok(self.entries.len())
    }
}

/// Words from a filename, for keyword matching alongside the vector.
fn filename_words(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    stem.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}
