//! What running an encoder in q8_0 actually costs.
//!
//! The browser cannot hold a 0.5B decoder as f32: 1.98GB of weights against a
//! 2GB cap on a single wasm32 allocation. Keeping weights block-quantised fixes
//! that, and every piece needed already exists: `LinearData::Q8_0`, quantised
//! matmul kernels on both targets, and `ModelLoadConfig::target_dtype`.
//!
//! What is not known is the price. Quantised matmul does more work per output
//! element than f32, so memory could fall 4x while latency rises. This measures
//! that on an encoder that already fits, which is far cheaper than rewriting the
//! container format and finding out afterwards.
//!
//! Run with `--nocapture`; it reports rather than asserting a threshold, because
//! the number is the point.

use std::path::PathBuf;
use std::time::Instant;

use kjarni_models::SentenceEncoder;
use kjarni_models::models::qwen::QwenModel;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::tensor::DType;
use kjarni_transformers::traits::Device;

fn model_dir() -> Option<PathBuf> {
    let d = kjarni_transformers::models::get_default_cache_dir()
        .join("sentence-transformers_all-MiniLM-L6-v2");
    d.join("model.safetensors").exists().then_some(d)
}

/// Encoders cannot run quantised, and this records why.
///
/// `LinearLayer::matmul` handles Q8_0, but the fused `matmul_noalloc` and
/// `weights_slice` paths panic with "Only f32 LinearLayer supported in optimized
/// path", and encoders go through those. This is not a problem worth fixing:
/// MiniLM is 23MB, it already fits everywhere, and f32 is the faster choice for
/// it. The note exists so nobody assumes the gap is an oversight.
#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "run with --release")]
async fn encoders_do_not_support_quantised_weights() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: minilm-l6-v2 not cached");
        return;
    };

    let cfg = ModelLoadConfig {
        target_dtype: Some(DType::Q8_0),
        ..Default::default()
    };
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        SentenceEncoder::from_pretrained(&dir, Device::Cpu, None, Some(cfg), None)
    }));

    match attempt {
        Err(_) => eprintln!("\n  encoder + Q8_0 panics in the fused path, as expected\n"),
        Ok(Err(e)) => eprintln!("\n  encoder + Q8_0 rejected: {e}\n"),
        Ok(Ok(_)) => eprintln!("\n  encoder + Q8_0 now loads; the fused path grew support\n"),
    }
}

/// Can a decoder run with quantised weights at all?
///
/// This is the question that decides the plan. The general `matmul` handles
/// Q8_0, but `matmul_noalloc` panics on it, and the decoder uses both. If the
/// fused path is reachable with quantised weights, keeping `.kjq` quantised
/// buys nothing until that path learns the other dtypes.
#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "run with --release")]
async fn can_a_decoder_run_quantised() {
    use kjarni_transformers::pipeline::DecoderLoader;

    let dir =
        kjarni_transformers::models::get_default_cache_dir().join("Qwen_Qwen2.5-0.5B-Instruct");
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: qwen2.5-0.5b not cached");
        return;
    }

    use kjarni_transformers::common::GenerationConfig;
    use kjarni_transformers::decoder::generator::DecoderGenerator;

    // Same prompt, same token budget, once per dtype.
    let mut report = vec![];
    for dtype in [None, Some(DType::Q8_0)] {
        let cfg = ModelLoadConfig {
            target_dtype: dtype,
            ..Default::default()
        };
        let model: QwenModel =
            match DecoderLoader::load_from_pretrained(&dir, Device::Cpu, None, Some(cfg), None) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  {dtype:?} failed to load: {e}");
                    continue;
                }
            };
        let generator = DecoderGenerator::new(std::sync::Arc::new(model)).expect("generator");
        let config = GenerationConfig {
            max_new_tokens: Some(16),
            ..Default::default()
        };

        let _ = generator
            .generate("Warm up the caches", &config, None)
            .await;

        let t = Instant::now();
        let out = generator
            .generate("The capital of Iceland is", &config, None)
            .await;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        report.push((format!("{dtype:?}"), ms, out.map(|o| o.trim().to_string())));
    }

    eprintln!("\n  16 tokens, Qwen2.5 0.5B on CPU");
    for (dtype, ms, out) in &report {
        eprintln!(
            "  {:<14} {:>8.1} ms   {:?}",
            dtype,
            ms,
            out.as_deref().unwrap_or("failed")
        );
    }
    if report.len() == 2 {
        eprintln!("  q8_0 is {:.2}x the f32 time\n", report[1].1 / report[0].1);
    }
}

/// The `.kjq` container must read both encodings, told apart by the magic.
///
/// KJQ1 files are already published on Hugging Face and cached on disk by the
/// Obsidian plugin, which only re-downloads when the file is missing. A format
/// change that could not read them would break every existing install silently.
#[test]
fn kjq_container_reads_both_encodings() {
    use kjarni_transformers::weights::kjq::{self, KjqEncoding};

    let dir = std::env::var("KJARNI_KJQ_DIR").unwrap_or_else(|_| "/tmp/kjq".into());
    let v1 = std::path::Path::new(&dir).join("all-MiniLM-L6-v2-q8.kjq");
    let v8 = std::path::Path::new(&dir).join("all-MiniLM-L6-v2-kjq8.kjq");

    if let Ok(bytes) = std::fs::read(&v1) {
        let u = kjq::unpack(&bytes).expect("KJQ1 must still unpack");
        assert_eq!(u.encoding, KjqEncoding::Kjq1);
        assert!(!u.config_json.is_empty() && !u.tokenizer_json.is_empty());
        eprintln!("  KJQ1: {} bytes of f32 safetensors", u.safetensors.len());
    } else {
        eprintln!("  skipping KJQ1: {} not present", v1.display());
    }

    if let Ok(bytes) = std::fs::read(&v8) {
        let u = kjq::unpack(&bytes).expect("KJQ8 must unpack");
        assert_eq!(u.encoding, KjqEncoding::Kjq8);
        assert!(!u.config_json.is_empty() && !u.tokenizer_json.is_empty());
        assert!(!u.blocks.is_empty(), "KJQ8 must yield block tensors");
        let total: usize = u.blocks.iter().map(|b| b.bytes.len()).sum();
        // 34 bytes per block: one f16 scale plus 32 int8 values.
        for b in &u.blocks {
            let expected = b.shape[0] * b.shape[1] / 32 * 34;
            assert_eq!(
                b.bytes.len(),
                expected,
                "{} has the wrong block length",
                b.name
            );
        }
        eprintln!("  KJQ8: {} block tensors, {total} bytes", u.blocks.len());
    } else {
        eprintln!("  skipping KJQ8: {} not present", v8.display());
    }
}

/// An unknown magic must be rejected, and say what it expected.
#[test]
fn kjq_rejects_an_unknown_magic() {
    use kjarni_transformers::weights::kjq;
    let err = kjq::unpack(b"NOPE\x00\x00\x00\x00")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("KJQ1") && err.contains("KJQ8"),
        "unhelpful error: {err}"
    );
}

/// A `KJQ8` decoder must load and generate without ever being expanded to f32.
///
/// This is the point of the whole encoding. Qwen2.5 0.5B is 494M parameters,
/// which is 1.98GB as f32 while wasm32 caps a single allocation at 2GB, so the
/// browser could load this model only if the weights stay in blocks. The engine
/// could already run quantised decoders; what was missing was a container that
/// did not undo it on the way in.
#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "generation is far too slow unoptimised")]
async fn kjq8_decoder_loads_and_generates() {
    use kjarni_transformers::common::GenerationConfig;
    use kjarni_transformers::decoder::generator::DecoderGenerator;
    use kjarni_transformers::pipeline::DecoderLoader;
    use kjarni_transformers::weights::kjq::{self, KjqEncoding};

    let dir = std::env::var("KJARNI_KJQ_DIR").unwrap_or_else(|_| "/tmp/kjq".into());
    // Named `-q8` though it carries KJQ8: that is the basename the published
    // path uses, and the fixture builder follows it.
    let path = std::path::Path::new(&dir).join("qwen05b-q8.kjq");
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipping: {} not present", path.display());
        eprintln!("  build it with: quantize_model.py --format kjq8");
        return;
    };

    let unpacked = kjq::unpack(&bytes).expect("unpack KJQ8");
    assert_eq!(unpacked.encoding, KjqEncoding::Kjq8);
    assert!(
        !unpacked.blocks.is_empty(),
        "a KJQ8 file must carry block tensors"
    );

    let model: QwenModel =
        DecoderLoader::load_from_kjq(&unpacked, None, None).expect("load qwen from KJQ8");

    let generator = DecoderGenerator::new(std::sync::Arc::new(model)).expect("generator");
    let config = GenerationConfig {
        max_new_tokens: Some(12),
        add_bos_token: false,
        ..Default::default()
    };

    let out = generator
        .generate("The capital of Iceland is", &config, None)
        .await
        .expect("generate from a KJQ8 model");

    let blocks: usize = unpacked.blocks.iter().map(|b| b.bytes.len()).sum();
    eprintln!(
        "\n  KJQ8 Qwen: {} block tensors, {:.1} MB of blocks",
        unpacked.blocks.len(),
        blocks as f64 / 1_048_576.0
    );
    eprintln!("  generated: {:?}\n", out.trim());

    // Coherence, not exact text. Block quantisation moves the weights, so the
    // tokens legitimately differ from f32, and this prompt continues as a quiz
    // rather than an answer in every configuration tested, f32 included. What
    // misread blocks would produce is garbage: random code points, or one token
    // repeating. Both are worth catching; the exact words are not.
    let text = out.trim();
    assert!(!text.is_empty(), "generated nothing");

    let printable = text
        .chars()
        .filter(|c| c.is_ascii_graphic() || c.is_whitespace())
        .count();
    let ratio = printable as f64 / text.chars().count() as f64;
    assert!(
        ratio > 0.95,
        "output is mostly non-ASCII, which is what misread blocks look like: {text:?}"
    );

    let words: Vec<&str> = text.split_whitespace().collect();
    let distinct = words.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        words.len() >= 3 && distinct >= words.len() / 2,
        "output is degenerate or repetitive, which is what a broken KV cache or \
         misread scale looks like: {text:?}"
    );
}
