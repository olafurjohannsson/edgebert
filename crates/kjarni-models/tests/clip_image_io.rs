//! Decoding real image files, and embedding straight from them.
//!
//! Separate binary from the parity suite because it needs the `image-io` feature.
//! The parity tests deliberately take raw pixels so they can run without it.

#![cfg(feature = "image-io")]

use kjarni_models::models::clip::{ClipVisionModel, decode_image_file};
use std::path::PathBuf;

fn fixtures() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("KJARNI_CLIP_FIXTURES").ok()?);
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// PNG decoding must reproduce the raw bytes exactly. It is lossless, so
/// anything other than an exact match is a bug in the channel handling.
#[test]
fn png_decodes_to_the_original_pixels() {
    let Some(fx) = fixtures() else {
        eprintln!("skipping: set KJARNI_CLIP_FIXTURES");
        return;
    };
    let raw = std::fs::read(fx.join("fixture.rgb8")).expect("raw rgb");
    let decoded = decode_image_file(&fx.join("fixture.png")).expect("decode png");

    assert_eq!(decoded.width, 320);
    assert_eq!(decoded.height, 240);
    assert_eq!(decoded.pixels, raw, "PNG is lossless; bytes must match");
}

/// Greyscale and RGBA both have to come out as three-channel RGB.
///
/// These are the shapes a real photo directory actually contains, and getting the
/// channel expansion wrong produces a valid-looking image with the colours in the
/// wrong places rather than an error.
#[test]
fn grayscale_and_rgba_become_rgb() {
    let Some(fx) = fixtures() else { return };

    for name in ["fixture_gray.png", "fixture_rgba.png"] {
        let path = fx.join(name);
        if !path.exists() {
            eprintln!("skipping {name}: not present");
            continue;
        }
        let img = decode_image_file(&path).expect("decode");
        assert_eq!(img.width, 320, "{name} width");
        assert_eq!(img.height, 240, "{name} height");
        assert_eq!(img.pixels.len(), 320 * 240 * 3, "{name} must be 3 channels");

        if name.starts_with("fixture_gray") {
            // Grey expands by replication, so all three channels agree.
            for px in img.pixels.chunks_exact(3).take(500) {
                assert_eq!(px[0], px[1], "grey channels must match");
                assert_eq!(px[1], px[2], "grey channels must match");
            }
        }
    }
}

/// Decoding is picked by magic bytes, so a JPEG named `.png` still works.
#[test]
fn format_is_detected_from_content_not_extension() {
    let Some(fx) = fixtures() else { return };
    let jpeg = std::fs::read(fx.join("fixture.jpg")).expect("jpeg bytes");
    let img = kjarni_models::models::clip::decode_image(&jpeg).expect("decode by content");
    assert_eq!((img.width, img.height), (320, 240));
}

/// Our JPEG decoder against PIL's, which is a different IDCT implementation.
///
/// Not byte-exact and should not be: JPEG's inverse DCT is defined to a tolerance,
/// not a bit pattern, so independent decoders legitimately differ by a step or two
/// per sample. What this rules out is a wrong colour transform or subsampling
/// handling, which show up as differences an order of magnitude larger.
#[test]
fn jpeg_decoding_agrees_with_an_independent_decoder() {
    let Some(fx) = fixtures() else { return };
    let reference = fx.join("fixture_jpeg_pil.rgb8");
    if !reference.exists() {
        eprintln!("skipping: no fixture_jpeg_pil.rgb8; rerun bench/clip_reference.py");
        return;
    }

    let ours = decode_image_file(&fx.join("fixture.jpg")).expect("decode jpeg");
    let theirs = std::fs::read(reference).expect("reference bytes");
    assert_eq!(ours.pixels.len(), theirs.len(), "decoded sizes differ");

    let mut worst = 0i32;
    let mut total = 0i64;
    for (a, b) in ours.pixels.iter().zip(&theirs) {
        let d = (*a as i32 - *b as i32).abs();
        worst = worst.max(d);
        total += d as i64;
    }
    let mean = total as f64 / theirs.len() as f64;
    println!("jpeg decode vs PIL: worst {worst}, mean {mean:.4}");
    assert!(
        worst <= 8,
        "worst sample differs by {worst}, expected IDCT-level noise"
    );
    assert!(
        mean < 1.0,
        "mean difference {mean} is too large for IDCT noise"
    );
}

/// The path a photo indexer actually takes: a file on disk to a vector.
///
/// The bar is agreement with HuggingFace on the *same* comparison rather than a
/// fixed similarity. JPEG loss on this fixture is severe, roughly 0.968, because
/// it is a checkerboard plus noise and that is close to the worst case for DCT
/// compression. A photograph would score far higher, and asserting a high number
/// here would be asserting something about the fixture, not about the code.
#[test]
fn embedding_from_a_jpeg_tracks_the_reference() {
    let (Some(fx), Some(dir)) = (fixtures(), model_dir()) else {
        eprintln!("skipping: fixtures or model missing");
        return;
    };
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fx.join("reference.json")).expect("ref"))
            .expect("parse reference");
    let Some(expected) = reference["png_vs_jpeg_cosine"].as_f64() else {
        eprintln!("skipping: reference.json predates png_vs_jpeg_cosine");
        return;
    };

    let model = ClipVisionModel::from_dir(&dir).expect("load vision tower");
    let from_png = model
        .embed_image_file(&fx.join("fixture.png"))
        .expect("embed png");
    let from_jpeg = model
        .embed_image_file(&fx.join("fixture.jpg"))
        .expect("embed jpeg");

    let cos = cosine(from_png.as_slice().unwrap(), from_jpeg.as_slice().unwrap());
    println!("png vs jpeg cosine: ours {cos:.6}, HuggingFace {expected:.6}");
    assert!(
        (cos as f64 - expected).abs() < 5e-3,
        "png/jpeg cosine drifted: ours {cos}, reference {expected}"
    );
}
