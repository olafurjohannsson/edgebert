//! Does the CLIP vision tower match HuggingFace?
//!
//! Deliberately two separate checks. Preprocessing and the transformer fail in
//! different ways, and a resize that is subtly wrong produces embeddings that look
//! entirely reasonable while ranking badly. Feeding HF's own `pixel_values` into
//! the tower isolates one from the other, so a failure says which half is wrong.
//!
//! Fixtures come from `bench/clip_reference.py`, which writes raw f32 arrays
//! beside a fixed PNG. Point `KJARNI_CLIP_FIXTURES` at that directory to run.

use kjarni_models::models::clip::{ClipConfig, ClipTextModel, ClipVisionModel};
use ndarray::Array3;
use std::path::PathBuf;

/// Committed fixtures, so these run with no environment set and no PyTorch.
///
/// `KJARNI_CLIP_FIXTURES` overrides, which is what `bench/clip_reference.py`
/// writes when regenerating.
fn fixtures() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("KJARNI_CLIP_FIXTURES") {
        let dir = PathBuf::from(dir);
        assert!(
            dir.is_dir(),
            "KJARNI_CLIP_FIXTURES is set to {} which is not a directory",
            dir.display()
        );
        return Some(dir);
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/clip");
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

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{} is not whole f32s",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// The tower alone, fed the exact tensor HF produced.
///
/// This is the assertion that matters: patch projection, class token, positions,
/// twelve pre-norm blocks, pooled class token, post-norm and projection, all
/// against a reference that used none of our code.
#[test]
fn vision_tower_matches_huggingface_on_reference_pixels() {
    let (Some(fx), Some(dir)) = (fixtures(), model_dir()) else {
        eprintln!("skipping: set KJARNI_CLIP_FIXTURES and download clip-vit-base-32");
        return;
    };

    let model = ClipVisionModel::from_dir(&dir).expect("load CLIP vision tower");
    let cfg = model.config();
    let edge = cfg.vision_config.image_size;

    let px = read_f32(&fx.join("pixel_values.f32"));
    assert_eq!(
        px.len(),
        3 * edge * edge,
        "reference pixel_values wrong size"
    );
    let pixels = Array3::from_shape_vec((3, edge, edge), px).expect("shape pixel_values");

    let got = model.embed_image_tensor(&pixels).expect("embed");
    let want = read_f32(&fx.join("image_embeds.f32"));

    assert_eq!(got.len(), want.len(), "embedding dimension differs");

    let cos = cosine(got.as_slice().unwrap(), &want);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("cosine {cos:.8}, worst element {worst:.8}");

    // Both, not just cosine: cosine alone would pass a uniformly scaled vector,
    // and these are L2-normalized on both sides so the scale should agree too.
    assert!(cos > 0.9999, "cosine to HuggingFace was only {cos}");
    assert!(worst < 1e-3, "worst element differs by {worst}");
}

/// Preprocessing alone, against the tensor HF's own processor produced.
///
/// Split out because this is the half most likely to drift: bicubic against
/// bilinear, or a half-pixel offset, changes every value a little and nothing
/// crashes.
#[test]
fn preprocessing_matches_huggingface() {
    let (Some(fx), Some(dir)) = (fixtures(), model_dir()) else {
        eprintln!("skipping: set KJARNI_CLIP_FIXTURES and download clip-vit-base-32");
        return;
    };

    let model = ClipVisionModel::from_dir(&dir).expect("load CLIP vision tower");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fx.join("fixture.json")).expect("meta"))
            .expect("parse meta");
    let w = meta["width"].as_u64().expect("width") as usize;
    let h = meta["height"].as_u64().expect("height") as usize;
    let rgb = std::fs::read(fx.join("fixture.rgb8")).expect("rgb bytes");

    let got = model
        .preprocessor()
        .preprocess_rgb8(&rgb, w, h)
        .expect("preprocess");
    let want = read_f32(&fx.join("pixel_values.f32"));

    assert_eq!(got.len(), want.len(), "preprocessed tensor size differs");
    let flat: Vec<f32> = got.iter().copied().collect();

    let worst = flat
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean = flat
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / flat.len() as f32;
    println!("preprocess: worst {worst:.6}, mean {mean:.6}");

    // The mean is the real assertion: measured at 7e-6, so anything at 1e-4 means
    // the pipeline changed. `worst` stays looser because a few pixels land on a
    // rounding boundary where our float filter and PIL's integer one disagree by
    // one 8-bit step, which normalization scales to about 0.015.
    assert!(mean < 1e-4, "mean preprocessed difference is {mean}");
    assert!(worst < 0.03, "worst preprocessed value differs by {worst}");
}

/// The whole path a caller actually uses: bytes in, vector out.
///
/// The two tests above can both pass while the pieces are wired together wrongly,
/// so this runs preprocessing and the tower back to back and checks the result
/// still lands on HuggingFace's embedding.
#[test]
fn embedding_raw_rgb_matches_huggingface() {
    let (Some(fx), Some(dir)) = (fixtures(), model_dir()) else {
        eprintln!("skipping: set KJARNI_CLIP_FIXTURES and download clip-vit-base-32");
        return;
    };

    let model = ClipVisionModel::from_dir(&dir).expect("load CLIP vision tower");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fx.join("fixture.json")).expect("meta"))
            .expect("parse meta");
    let w = meta["width"].as_u64().expect("width") as usize;
    let h = meta["height"].as_u64().expect("height") as usize;
    let rgb = std::fs::read(fx.join("fixture.rgb8")).expect("rgb bytes");

    let got = model.embed_rgb8(&rgb, w, h).expect("embed image");
    let want = read_f32(&fx.join("image_embeds.f32"));

    let cos = cosine(got.as_slice().unwrap(), &want);
    println!("end to end cosine {cos:.8}");

    // Retrieval only cares about direction, and preprocessing rounding costs a
    // little more than the tower alone does.
    assert!(cos > 0.9995, "end-to-end cosine was only {cos}");

    // A normalized vector, because that is what the index will assume.
    let norm: f32 = got.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "embedding norm was {norm}");
}

/// The text tower against HuggingFace, on token ids HF itself produced.
///
/// Ids come from the reference rather than from our tokenizer on purpose: this
/// asks whether the transformer is right, not whether the BPE is. Mixing the two
/// would leave a failure ambiguous, the same reason preprocessing is split from
/// the vision tower above.
#[test]
fn text_tower_matches_huggingface() {
    let (Some(fx), Some(dir)) = (fixtures(), model_dir()) else {
        eprintln!("skipping: set KJARNI_CLIP_FIXTURES and download clip-vit-base-32");
        return;
    };
    let ref_path = fx.join("text_reference.json");
    if !ref_path.exists() {
        eprintln!("skipping: no text_reference.json; rerun bench/clip_reference.py");
        return;
    }

    let config = ClipConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("config.json"),
    )
    .expect("parse config");
    let weights = kjarni_transformers::weights::ModelWeights::new(&dir).expect("load weights");
    let model = ClipTextModel::from_weights(&config, &weights).expect("build text tower");

    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&ref_path).expect("read reference"))
            .expect("parse reference");

    for (i, case) in cases.as_array().expect("array").iter().enumerate() {
        let text = case["text"].as_str().expect("text");
        let ids: Vec<u32> = case["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let eos = case["eos_index"].as_u64().expect("eos_index") as usize;

        let got = model.embed_ids(&ids, eos).expect("embed text");
        let want = read_f32(&fx.join(format!("text_{i}.f32")));

        let cos = cosine(got.as_slice().unwrap(), &want);
        println!("  {text:?}: cosine {cos:.8}");
        assert!(cos > 0.9999, "text {text:?} had cosine {cos}");
    }
}

/// A caption and a matching image should land near each other, and further from
/// an unrelated caption.
///
/// The towers can each match their own reference while being wired into different
/// spaces, and nothing above would catch that. This is the check that the shared
/// space is actually shared.
#[test]
fn image_and_text_land_in_the_same_space() {
    let (Some(fx), Some(dir)) = (fixtures(), model_dir()) else {
        eprintln!("skipping: set KJARNI_CLIP_FIXTURES and download clip-vit-base-32");
        return;
    };
    if !fx.join("text_reference.json").exists() {
        eprintln!("skipping: no text_reference.json");
        return;
    }

    let image = read_f32(&fx.join("image_embeds.f32"));
    let text0 = read_f32(&fx.join("text_0.f32"));

    let vision = ClipVisionModel::from_dir(&dir).expect("load vision tower");
    let px = read_f32(&fx.join("pixel_values.f32"));
    let edge = vision.config().vision_config.image_size;
    let pixels = ndarray::Array3::from_shape_vec((3, edge, edge), px).expect("pixels");
    let ours = vision.embed_image_tensor(&pixels).expect("embed image");

    // Our image vector against HF's text vector. If the towers were projected into
    // different bases this would be near zero however good each tower was on its
    // own; CLIP similarities are small in absolute terms, so the bar is the
    // agreement with HF's own cross-modal score rather than a fixed threshold.
    let theirs = cosine(&image, &text0);
    let mixed = cosine(ours.as_slice().unwrap(), &text0);
    println!("cross-modal cosine: HF {theirs:.6}, ours {mixed:.6}");
    assert!(
        (theirs - mixed).abs() < 1e-3,
        "cross-modal similarity drifted: HF {theirs}, ours {mixed}"
    );
}

/// Our tokenizer against HuggingFace's, id for id.
///
/// Exact equality, not a similarity score: one wrong id is a different word, and
/// the embedding it produces will look perfectly reasonable while meaning
/// something else. Cases cover the things that actually differ between BPE
/// implementations rather than just a happy-path caption.
#[test]
fn tokenizer_matches_huggingface() {
    let Some(fx) = fixtures() else {
        eprintln!("skipping: set KJARNI_CLIP_FIXTURES");
        return;
    };
    let Some(dir) = model_dir() else { return };
    let ref_path = fx.join("tokenizer_reference.json");
    if !ref_path.exists() {
        eprintln!("skipping: no tokenizer_reference.json; rerun bench/clip_reference.py");
        return;
    }

    let tokenizer = kjarni_models::models::clip::ClipTokenizer::from_json_str(
        &std::fs::read_to_string(dir.join("tokenizer.json")).expect("tokenizer.json"),
    )
    .expect("build tokenizer");

    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&ref_path).expect("read")).expect("parse");

    let mut failures = Vec::new();
    for case in cases.as_array().expect("array") {
        let text = case["text"].as_str().expect("text");
        let want: Vec<u32> = case["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let got = tokenizer.encode(text);
        if got != want {
            failures.push(format!("  {text:?}\n    want {want:?}\n    got  {got:?}"));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} cases disagree with HuggingFace:\n{}",
        failures.len(),
        cases.as_array().unwrap().len(),
        failures.join("\n")
    );
}

/// Text in, image in, one comparable number out.
///
/// Every other test here is fed something HuggingFace produced: reference pixels,
/// reference token ids. This one starts from a caption string and raw image bytes
/// and uses nothing but our own code, which is the only way to know the pieces
/// are wired together rather than merely correct in isolation.
#[test]
fn a_caption_can_be_compared_to_an_image_end_to_end() {
    let (Some(fx), Some(dir)) = (fixtures(), model_dir()) else {
        eprintln!("skipping: set KJARNI_CLIP_FIXTURES and download clip-vit-base-32");
        return;
    };
    if !fx.join("text_reference.json").exists() {
        eprintln!("skipping: no text_reference.json");
        return;
    }

    let config = ClipConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("config.json"),
    )
    .expect("parse config");
    let weights = kjarni_transformers::weights::ModelWeights::new(&dir).expect("weights");
    let text_model = ClipTextModel::from_weights(&config, &weights).expect("text tower");
    let tokenizer = kjarni_models::models::clip::ClipTokenizer::from_json_str(
        &std::fs::read_to_string(dir.join("tokenizer.json")).expect("tokenizer.json"),
    )
    .expect("tokenizer");
    let vision = ClipVisionModel::from_dir(&dir).expect("vision tower");

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fx.join("fixture.json")).expect("meta"))
            .expect("parse meta");
    let w = meta["width"].as_u64().expect("width") as usize;
    let h = meta["height"].as_u64().expect("height") as usize;
    let rgb = std::fs::read(fx.join("fixture.rgb8")).expect("rgb");
    let image = vision.embed_rgb8(&rgb, w, h).expect("embed image");

    let embed = |caption: &str| {
        let ids = tokenizer.encode(caption);
        text_model
            .embed_ids(&ids, ids.len() - 1)
            .expect("embed caption")
    };

    // The fixture is a synthetic gradient with a magenta block and a checkerboard,
    // so no caption is a good description of it. What must hold is that our own
    // score agrees with HuggingFace's for the same caption, which is what a
    // retrieval ranking is built out of.
    let ours = embed("a photo of a dog");
    let reference = read_f32(&fx.join("text_0.f32"));
    let cos_text = cosine(ours.as_slice().unwrap(), &reference);
    println!("caption tokenized and embedded by us vs HuggingFace: {cos_text:.8}");
    assert!(
        cos_text > 0.9999,
        "our own caption pipeline drifted: {cos_text}"
    );

    let theirs_cross = cosine(&read_f32(&fx.join("image_embeds.f32")), &reference);
    let ours_cross = cosine(image.as_slice().unwrap(), ours.as_slice().unwrap());
    println!("cross-modal: HF {theirs_cross:.6}, ours {ours_cross:.6}");
    assert!(
        (theirs_cross - ours_cross).abs() < 1e-3,
        "cross-modal score drifted: HF {theirs_cross}, ours {ours_cross}"
    );
}
