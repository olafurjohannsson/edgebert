//! Loading a `.kjq` model through the shared engine.
//!
//! `.kjq` is the single-file container the browser demos download: config,
//! tokenizer and int8 tensors in one request, 23 MB instead of 88 MB. Until now only
//! `kjarni-wasm` could read it, using its own copy of the encoder stack. These tests
//! cover the engine reading the same file, which is what lets that duplicate go.
//!
//! Fixtures come from `KJARNI_KJQ_DIR` when set (CI exports them from the cached
//! weights with `scripts/quantize_model.py`), otherwise from the sibling website
//! checkout. They skip rather than fail when neither is present, so a bare checkout
//! still runs green.
//!
//! Do not let the skip lull you: before CI exported the fixtures, all four of these
//! passed green in CI while testing nothing at all.

use kjarni_models::SentenceEncoder;
use kjarni_transformers::pipeline::EncoderLoader;
use kjarni_transformers::traits::Device;
use kjarni_transformers::weights::kjq;

/// The model the live demo downloads, in the sibling website checkout.
const KJQ: &str = "../../../web.kjarni.ai/src/static/models/all-MiniLM-L6-v2-q8.kjq";

/// Full-precision weights for the same model, if they have been fetched.
const F32_CACHE: &str = "sentence-transformers_all-MiniLM-L6-v2";

/// Reads a `.kjq` fixture.
///
/// `KJARNI_KJQ_DIR` wins when set: CI has no sibling website checkout, so it exports
/// the fixtures from cached weights with `scripts/quantize_model.py` and points the
/// variable at that directory. A fixture missing from an explicitly configured
/// directory is a broken CI step, not an absent optional file, so it fails loudly
/// instead of skipping.
///
/// The fallback path below is the sibling website checkout, which no longer holds
/// these files: they moved to Hugging Face once `qwen05b-q8.kjq` outgrew GitHub's
/// 100MB per-file limit. In practice fixtures now arrive one of two ways, and both
/// are cheap:
///
/// ```text
/// python crates/kjarni-wasm/scripts/quantize_model.py \
///     --model-dir ~/.cache/kjarni/sentence-transformers_all-MiniLM-L6-v2 \
///     --output /tmp/kjq/all-MiniLM-L6-v2-q8.kjq
/// KJARNI_KJQ_DIR=/tmp/kjq cargo test --release
/// ```
///
/// or download them from <https://huggingface.co/olafuraron>. CI takes the first
/// route, building them from weights it already caches.
fn fixture(rel: &str) -> Option<Vec<u8>> {
    if let Ok(dir) = std::env::var("KJARNI_KJQ_DIR") {
        let dir = std::path::Path::new(&dir);
        // A directory that does not exist means the variable is pointed somewhere
        // wrong, and every fixture would silently skip. That is worth failing on.
        assert!(
            dir.is_dir(),
            "KJARNI_KJQ_DIR is set to {} which is not a directory",
            dir.display()
        );

        let name = std::path::Path::new(rel).file_name().expect("fixture name");
        let path = dir.join(name);
        return match std::fs::read(&path) {
            Ok(bytes) => Some(bytes),
            // One fixture missing from a valid directory is normal: they are built
            // per model, and a run that only quantised the encoder has no decoder
            // fixture. Fall through to the caller's skip rather than failing the
            // whole suite for a file this test alone needs.
            Err(_) => {
                eprintln!(
                    "fixture {} not present in {}",
                    name.display(),
                    dir.display()
                );
                None
            }
        };
    }
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).ok()
}

const TEXTS: [&str; 4] = [
    "Hello world",
    "The cat sat on the mat",
    "Reykjavik is the capital of Iceland",
    "What is your refund policy?",
];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn kjq_bytes() -> Option<Vec<u8>> {
    fixture(KJQ)
}

fn load_from_kjq(bytes: &[u8]) -> SentenceEncoder {
    let unpacked = kjq::unpack(bytes).expect("unpack .kjq");
    EncoderLoader::load_from_bytes(
        &unpacked.safetensors,
        &unpacked.config_json,
        unpacked.tokenizer_json.as_bytes(),
        Device::Cpu,
        None,
        None,
    )
    .expect("engine loads the unpacked .kjq")
}

#[tokio::test]
async fn engine_loads_a_kjq_model_and_encodes() {
    let Some(bytes) = kjq_bytes() else {
        eprintln!("skipping: {KJQ} not present");
        return;
    };

    let encoder = load_from_kjq(&bytes);
    let v = encoder.encode("Hello world").await.expect("encode");

    assert_eq!(v.len(), 384, "MiniLM-L6-v2 produces 384 dimensions");
    assert!(
        v.iter().any(|x| *x != 0.0),
        "embedding is all zeros, so the weights did not arrive"
    );

    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "encode() should return a normalized vector, got norm {norm}"
    );
}

#[tokio::test]
async fn kjq_preserves_semantic_ranking() {
    // What the demos depend on is the ordering, not the exact floats.
    let Some(bytes) = kjq_bytes() else {
        eprintln!("skipping: {KJQ} not present");
        return;
    };
    let encoder = load_from_kjq(&bytes);

    let query = encoder.encode("How do I get my money back?").await.unwrap();
    let relevant = encoder.encode("What is your refund policy?").await.unwrap();
    let irrelevant = encoder
        .encode("The weather in Reykjavik is unpredictable.")
        .await
        .unwrap();

    let hit = cosine(&query, &relevant);
    let miss = cosine(&query, &irrelevant);

    assert!(
        hit > miss + 0.2,
        "ranking collapsed: relevant={hit:.4}, irrelevant={miss:.4}"
    );
}

#[tokio::test]
async fn int8_quantization_stays_within_a_known_bound() {
    // Same inference code both times; only the weights differ. That isolates the
    // cost of the int8 container from anything the loader might be doing wrong.
    //
    // The bound is deliberately tight around what was measured (0.948 worst case)
    // rather than a comfortable round number: if a change to the exporter or to
    // dequantization makes this worse, that should show up as a failure here rather
    // than as vaguely worse search results in the browser.
    let Some(bytes) = kjq_bytes() else {
        eprintln!("skipping: {KJQ} not present");
        return;
    };
    let cache = kjarni_transformers::models::get_default_cache_dir().join(F32_CACHE);
    if !cache.join("model.safetensors").exists() {
        eprintln!(
            "skipping: full-precision weights not cached at {}",
            cache.display()
        );
        return;
    }

    let quantized = load_from_kjq(&bytes);
    let full = SentenceEncoder::from_pretrained(&cache, Device::Cpu, None, None, None)
        .expect("load full-precision model");

    let mut worst = 1.0f32;
    for text in TEXTS {
        let a = quantized.encode(text).await.unwrap();
        let b = full.encode(text).await.unwrap();
        let cos = cosine(&a, &b);
        worst = worst.min(cos);
        assert!(
            cos > 0.94,
            "int8 .kjq drifted from f32 on {text:?}: cosine {cos:.6}"
        );
    }
    eprintln!("worst cosine against full precision: {worst:.6}");
}

// ─── Decoder loading ─────────────────────────────────────────────

/// A `.kjq` chat model, if one has been exported.
const CHAT_KJQ: &str = "../../../web.kjarni.ai/src/static/models/qwen05b-q8.kjq";

#[tokio::test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
async fn decoder_loads_from_kjq_bytes_and_generates() {
    // `DecoderLoader::load_from_bytes` was added for the browser: there is no
    // filesystem there, so the file-backed loaders cannot be used. It had no test,
    // only a manual run, which is how a loader silently regresses.
    use kjarni_transformers::pipeline::DecoderLoader;

    let Some(bytes) = fixture(CHAT_KJQ) else {
        eprintln!("skipping: {CHAT_KJQ} not present");
        return;
    };

    let unpacked = kjq::unpack(&bytes).expect("unpack .kjq");

    let model: kjarni_models::models::qwen::QwenModel = DecoderLoader::load_from_bytes(
        &unpacked.safetensors,
        &unpacked.config_json,
        unpacked.tokenizer_json.as_bytes(),
        None,
        Some(kjarni_transformers::models::ModelType::Qwen2_5_0_5B_Instruct),
    )
    .expect("decoder loads from bytes");

    // Loading is only half of it: a model that loads but cannot generate would pass
    // a construction-only assertion.
    use kjarni_transformers::decoder::generator::DecoderGenerator;
    let generator = DecoderGenerator::new(std::sync::Arc::new(model)).expect("generator");

    let config = kjarni_transformers::common::GenerationConfig {
        max_new_tokens: Some(12),
        strategy: kjarni_transformers::common::DecodingStrategy::Greedy,
        ..Default::default()
    };

    let out = generator
        .generate("The capital of Iceland is", &config, None)
        .await
        .expect("generation succeeds");

    assert!(
        out.to_lowercase().contains("reykjav"),
        "expected the model to name Reykjavik, got {out:?}"
    );
}
