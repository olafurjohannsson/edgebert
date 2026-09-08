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
use rayon::prelude::*;
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
    /// Extensions that look like images but have no decoder, and how many of each.
    ///
    /// Separate from `failures` because these never reach a decoder: the loader
    /// filters them out by extension. Counted because otherwise they vanish. An
    /// iPhone library is mostly HEIC, so pointing this at one and reporting only
    /// the JPEGs would silently index about half of it.
    pub unsupported: std::collections::BTreeMap<String, usize>,
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
        self.insert(IndexedImage {
            path: path.to_path_buf(),
            embedding: embedding.to_vec(),
            filename_text: filename_words(path),
        });
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

        let paths: Vec<PathBuf> = chunks
            .into_iter()
            .filter(|c| c.metadata.custom.get(KIND_KEY).map(String::as_str) == Some(KIND_IMAGE))
            .filter_map(|c| c.metadata.source.map(PathBuf::from))
            .collect();

        // One image at a time left 23 of 24 cores idle: a single 224x224 image is
        // 50 positions through 12 blocks, which is far too little work for the
        // encoder's own parallelism to fill a machine. Decoding, preprocessing and
        // embedding are independent per image, so the scan parallelises across
        // files instead of within one.
        //
        // Insertion stays serial and in path order, so a rescan produces the same
        // index whatever order the threads finished in.
        let embedded: Vec<(PathBuf, Result<Vec<f32>>)> = paths
            .into_par_iter()
            .map(|path| {
                let vector = self
                    .vision
                    .embed_image_file(&path)
                    .with_context(|| format!("embedding {}", path.display()))
                    .map(|v| v.to_vec());
                (path, vector)
            })
            .collect();

        let mut report = ScanReport {
            unsupported: unsupported_image_files(dir),
            ..Default::default()
        };

        for (path, vector) in embedded {
            match vector {
                Ok(embedding) => {
                    self.insert(IndexedImage {
                        filename_text: filename_words(&path),
                        path,
                        embedding,
                    });
                    report.added += 1;
                }
                Err(e) => {
                    report.skipped += 1;
                    report.failures.push((path, format!("{e:#}")));
                }
            }
        }
        Ok(report)
    }

    /// Adds or replaces by path, so rescanning a directory updates rather than
    /// duplicating.
    fn insert(&mut self, entry: IndexedImage) {
        match self.by_path.get(&entry.path) {
            Some(&at) => self.entries[at] = entry,
            None => {
                self.by_path.insert(entry.path.clone(), self.entries.len());
                self.entries.push(entry);
            }
        }
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

/// Counts files that look like photos but have no decoder here.
///
/// Extension-based, matching how the loader decides what to walk. The list is the
/// formats a real photo library actually contains, so a user is told what was left
/// out rather than quietly getting a partial index.
fn unsupported_image_files(dir: &Path) -> std::collections::BTreeMap<String, usize> {
    const LOOKS_LIKE_AN_IMAGE: &[&str] = &[
        "heic", "heif", "webp", "avif", "tif", "tiff", "bmp", "gif", "raw", "dng", "cr2", "cr3",
        "nef", "arw", "orf", "rw2", "raf",
    ];

    let mut counts = std::collections::BTreeMap::new();
    let Ok(entries) = walkdir_files(dir) else {
        return counts;
    };
    for path in entries {
        let Some(ext) = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
        else {
            continue;
        };
        if LOOKS_LIKE_AN_IMAGE.contains(&ext.as_str()) {
            *counts.entry(ext).or_insert(0) += 1;
        }
    }
    counts
}

/// Every file under `dir`, recursively.
fn walkdir_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    Ok(out)
}

/// Words from a filename, for keyword matching alongside the vector.
fn filename_words(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let words: Vec<&str> = stem
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        // Fall back to the raw filename rather than indexing an image with no
        // searchable text at all. `DocumentLoader` does the same, and the two
        // used to disagree on exactly this case.
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string()
    } else {
        words.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Filenames are often the only words a photo has, so they are worth keeping
    /// alongside the vector for keyword matching. This is not a caption and does
    /// not pretend to be one.
    #[test]
    fn filenames_become_searchable_words() {
        for (path, want) in [
            (
                "/p/IMG_2024-08-14_beach_sunset.jpg",
                "IMG 2024 08 14 beach sunset",
            ),
            ("/p/holiday.png", "holiday"),
            ("/p/DSC00123.JPG", "DSC00123"),
            ("/p/a.b.c/photo-01.jpeg", "photo 01"),
        ] {
            assert_eq!(filename_words(Path::new(path)), want, "for {path}");
        }
    }

    /// A name with nothing word-like must not become an empty string, or the
    /// image is indexed with no searchable text at all. Falls back to the raw
    /// filename, which is what `DocumentLoader` does for the same case.
    #[test]
    fn a_nameless_file_falls_back_to_the_filename() {
        assert_eq!(filename_words(Path::new("/p/___.png")), "___.png");
        assert_eq!(filename_words(Path::new("/p/.hidden")), "hidden");
        assert_eq!(filename_words(Path::new("/p/---.jpg")), "---.jpg");
    }

    /// Round-tripping an index must preserve the vectors exactly. A float that
    /// drifts through JSON would change every score by a little, which is the kind
    /// of thing that never fails and quietly degrades ranking.
    #[test]
    fn entries_survive_a_json_round_trip() {
        let entries = vec![
            IndexedImage {
                path: PathBuf::from("/photos/beach.jpg"),
                embedding: vec![0.1, -0.25, 0.333_333_34, 1.0, -1.0],
                filename_text: "beach".to_string(),
            },
            IndexedImage {
                path: PathBuf::from("/photos/a b/c.png"),
                embedding: vec![0.0; 5],
                filename_text: "c".to_string(),
            },
        ];

        let json = serde_json::to_vec(&entries).expect("serialise");
        let back: Vec<IndexedImage> = serde_json::from_slice(&json).expect("parse");

        assert_eq!(back.len(), entries.len());
        for (a, b) in entries.iter().zip(&back) {
            assert_eq!(a.path, b.path, "paths with spaces must survive");
            assert_eq!(a.filename_text, b.filename_text);
            assert_eq!(a.embedding, b.embedding, "vectors must be bit-identical");
        }
    }

    /// The scan report has to distinguish "indexed nothing" from "failed at
    /// everything", or a directory of corrupt files looks like an empty one.
    #[test]
    fn a_fresh_report_counts_nothing() {
        let r = ScanReport::default();
        assert_eq!(r.added, 0);
        assert_eq!(r.skipped, 0);
        assert!(r.failures.is_empty());
    }
}
