//! Searching a directory of photos by typing a description.
//!
//! The point of the whole CLIP module, end to end: index files on disk, type a
//! sentence, get the right picture back. Nothing here is fed anything a reference
//! implementation produced.

#![cfg(feature = "image-io")]

use kjarni_models::models::clip::ImageIndex;
use std::path::PathBuf;

fn photos() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("KJARNI_CLIP_PHOTOS").ok()?);
    dir.is_dir().then_some(dir)
}

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("KJARNI_CLIP_MODEL")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let home = std::env::var("HOME").ok()?;
            Some(PathBuf::from(home).join(".cache/kjarni/laion_CLIP-ViT-B-32-laion2B-s34B-b79K"))
        })?;
    dir.is_dir().then_some(dir)
}

fn index() -> Option<ImageIndex> {
    let (dir, model) = (photos()?, model_dir()?);
    let mut idx = ImageIndex::load(&model).expect("load CLIP");
    let report = idx.add_directory(&dir).expect("scan photos");
    assert!(
        report.failures.is_empty(),
        "failed to index: {:?}",
        report.failures
    );
    assert!(
        report.added >= 4,
        "expected at least 4 photos, got {}",
        report.added
    );
    Some(idx)
}

/// The headline claim: a description finds the right photo.
///
/// Ranking, not absolute score. CLIP cosines are small even for a strong match,
/// so what matters is that the intended image comes first.
#[test]
fn a_description_finds_the_right_photo() {
    let Some(idx) = index() else {
        eprintln!("skipping: set KJARNI_CLIP_PHOTOS and download clip-vit-base-32");
        return;
    };

    let cases = [
        ("a sandy beach under a blue sky", "beach_holiday"),
        ("a dense green forest with trees", "deep_forest"),
        ("a red car", "red_car"),
        ("a snowy mountain peak", "snowy_mountain"),
    ];

    let mut wrong = Vec::new();
    for (query, expected) in cases {
        let hits = idx.search(query, 4).expect("search");
        let top = hits[0]
            .path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        println!(
            "  {query:?} -> {top} ({:.4})  [{}]",
            hits[0].score,
            hits.iter()
                .map(|h| format!(
                    "{}:{:.3}",
                    h.path.file_stem().unwrap().to_string_lossy(),
                    h.score
                ))
                .collect::<Vec<_>>()
                .join(" ")
        );
        if top != expected {
            wrong.push(format!("{query:?} ranked {top} above {expected}"));
        }
    }
    assert!(wrong.is_empty(), "wrong top hit:\n  {}", wrong.join("\n  "));
}

/// Rescanning must update in place rather than duplicate.
///
/// A photo index gets rebuilt whenever a directory changes, so this is the normal
/// case and not an edge one.
#[test]
fn rescanning_does_not_duplicate() {
    let (Some(dir), Some(model)) = (photos(), model_dir()) else {
        return;
    };
    let mut idx = ImageIndex::load(&model).expect("load");

    let first = idx.add_directory(&dir).expect("scan").added;
    let before = idx.len();
    idx.add_directory(&dir).expect("rescan");

    assert_eq!(idx.len(), before, "rescanning grew the index from {before}");
    assert_eq!(idx.len(), first, "index size should match files scanned");
}

/// Vectors survive a save and load, and searching still returns the same answer.
#[test]
fn a_saved_index_reloads_and_still_searches() {
    let Some(idx) = index() else { return };
    let Some(model) = model_dir() else { return };

    let file = std::env::temp_dir().join("kjarni-clip-index-test.json");
    idx.save(&file).expect("save");

    let mut reloaded = ImageIndex::load(&model).expect("load models");
    let n = reloaded.load_entries(&file).expect("load entries");
    assert_eq!(n, idx.len(), "entry count changed across save/load");

    let q = "a red car";
    let a = idx.search(q, 1).expect("search original");
    let b = reloaded.search(q, 1).expect("search reloaded");
    assert_eq!(a[0].path, b[0].path, "reloaded index ranked differently");
    assert!((a[0].score - b[0].score).abs() < 1e-6, "score drifted");

    let _ = std::fs::remove_file(&file);
}

/// An index built with a different checkpoint must be refused, not silently
/// searched against vectors it cannot compare with.
#[test]
fn an_index_with_the_wrong_dimension_is_rejected() {
    let Some(model) = model_dir() else { return };
    let mut idx = ImageIndex::load(&model).expect("load");

    let file = std::env::temp_dir().join("kjarni-clip-bad-dim.json");
    std::fs::write(
        &file,
        serde_json::json!([{
            "path": "/tmp/whatever.jpg",
            "embedding": vec![0.1f32; 768],
            "filename_text": "whatever"
        }])
        .to_string(),
    )
    .expect("write");

    let err = idx.load_entries(&file).expect_err("must reject");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("768"),
        "error should name the bad dimension: {msg}"
    );
    let _ = std::fs::remove_file(&file);
}
