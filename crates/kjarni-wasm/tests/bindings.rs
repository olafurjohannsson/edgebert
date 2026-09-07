//! Behaviour tests for the browser bindings.
//!
//! This crate had no tests at all until now: it was `#![cfg(target_arch = "wasm32")]`
//! *and* excluded from the workspace, so nothing could reach it. That is how it came
//! to carry 1,250 lines of duplicate encoder code that quietly drifted from the
//! engine. The bindings still only export to JavaScript on wasm32, but the crate
//! builds natively, so what they do can be checked here on every `cargo test`.
//!
//! Tests skip when the model files are absent so a checkout without the website
//! repo alongside it still runs green.
//!
//! Only success paths are exercised here. Every binding reports failure as a
//! `JsValue`, and constructing one panics outside wasm32, so an assertion like
//! "rejects a file that is not .kjq" cannot run on the host. That behaviour is
//! covered a layer down instead, in `kjarni_transformers::weights::kjq`, which is
//! plain Rust and has its own tests for bad magic bytes and truncation.

use kjarni_wasm::{WasmClassifier, WasmModel, WasmReranker};

const EMBED_KJQ: &str = "../../../web.kjarni.ai/src/static/models/all-MiniLM-L6-v2-q8.kjq";
const RERANK_KJQ: &str = "../../../web.kjarni.ai/src/static/models/ms-marco-MiniLM-L-6-v2-q8.kjq";
const CLASSIFY_KJQ: &str = "../../../web.kjarni.ai/src/static/models/distilbert-sentiment-q8.kjq";

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
fn model_bytes(rel: &str) -> Option<Vec<u8>> {
    if let Ok(dir) = std::env::var("KJARNI_KJQ_DIR") {
        let name = std::path::Path::new(rel).file_name().expect("fixture name");
        let path = std::path::Path::new(&dir).join(name);
        return match std::fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(e) => panic!(
                "KJARNI_KJQ_DIR is set but {} could not be read: {e}",
                path.display()
            ),
        };
    }
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).ok()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// `encode` returns every vector concatenated, so callers slice by dimension.
fn split(flat: &[f32], count: usize) -> Vec<&[f32]> {
    let dim = flat.len() / count;
    (0..count).map(|i| &flat[i * dim..(i + 1) * dim]).collect()
}

// ─── WasmModel ───────────────────────────────────────────────────

#[test]
fn model_loads_from_kjq_and_encodes() {
    let Some(bytes) = model_bytes(EMBED_KJQ) else {
        eprintln!("skipping: {EMBED_KJQ} not present");
        return;
    };
    let model = WasmModel::from_quantized(&bytes).expect("load .kjq");

    let flat = model
        .encode(vec!["Hello world".into()], true)
        .expect("encode");

    assert_eq!(flat.len(), 384, "MiniLM-L6-v2 produces 384 dimensions");
    assert!(flat.iter().any(|v| *v != 0.0), "all zeros: weights missing");
}

#[test]
fn model_flattens_a_batch_in_input_order() {
    // The demo relies on this: it slices the flat array back into per-sentence
    // vectors by index, so order is part of the contract.
    let Some(bytes) = model_bytes(EMBED_KJQ) else {
        return;
    };
    let model = WasmModel::from_quantized(&bytes).unwrap();

    let texts: Vec<String> = vec!["cat".into(), "dog".into(), "quantum physics".into()];
    let flat = model.encode(texts.clone(), true).unwrap();
    assert_eq!(flat.len(), 384 * 3);

    let batched = split(&flat, 3);
    for (i, text) in texts.iter().enumerate() {
        let alone = model.encode(vec![text.clone()], true).unwrap();
        let cos = cosine(batched[i], &alone);
        assert!(
            cos > 0.999,
            "batch position {i} does not match encoding {text:?} alone: cosine {cos:.5}"
        );
    }
}

#[test]
fn model_normalization_flag_is_honoured() {
    let Some(bytes) = model_bytes(EMBED_KJQ) else {
        return;
    };
    let model = WasmModel::from_quantized(&bytes).unwrap();

    let normalized = model.encode(vec!["Hello world".into()], true).unwrap();
    let raw = model.encode(vec!["Hello world".into()], false).unwrap();

    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm(&normalized) - 1.0).abs() < 1e-3,
        "normalize=true should give unit length"
    );
    assert!(
        (norm(&raw) - 1.0).abs() > 1e-3,
        "normalize=false should not be unit length"
    );
}

#[test]
fn model_ranks_by_meaning_not_shared_words() {
    let Some(bytes) = model_bytes(EMBED_KJQ) else {
        return;
    };
    let model = WasmModel::from_quantized(&bytes).unwrap();

    let flat = model
        .encode(
            vec![
                "How do I get my money back?".into(),
                "What is your refund policy?".into(),
                "The weather in Reykjavik is unpredictable.".into(),
            ],
            true,
        )
        .unwrap();
    let v = split(&flat, 3);

    assert!(
        cosine(v[0], v[1]) > cosine(v[0], v[2]) + 0.2,
        "the refund sentence should rank far above the weather one"
    );
}

// ─── WasmReranker ────────────────────────────────────────────────

#[test]
fn reranker_puts_the_paraphrase_above_the_keyword_decoy() {
    // This is the example the demo ships and the blog post describes: the winner
    // contains neither "cancel" nor "subscription", and the decoy contains both.
    let Some(bytes) = model_bytes(RERANK_KJQ) else {
        eprintln!("skipping: {RERANK_KJQ} not present");
        return;
    };
    let reranker = WasmReranker::load(&bytes).expect("load cross-encoder");

    let paraphrase = reranker
        .score(
            "how do I cancel my subscription",
            "To end your plan, go to Settings and choose Close account.",
        )
        .unwrap();
    let decoy = reranker
        .score(
            "how do I cancel my subscription",
            "Cancellation of orders is handled by the warehouse team.",
        )
        .unwrap();

    assert!(
        paraphrase > decoy,
        "keyword decoy outranked the paraphrase: {paraphrase:.3} vs {decoy:.3}"
    );
}

#[test]
fn reranker_scores_are_deterministic() {
    let Some(bytes) = model_bytes(RERANK_KJQ) else {
        return;
    };
    let reranker = WasmReranker::load(&bytes).unwrap();

    let a = reranker
        .score("capital of Iceland", "Reykjavik is the capital.")
        .unwrap();
    let b = reranker
        .score("capital of Iceland", "Reykjavik is the capital.")
        .unwrap();
    assert_eq!(a, b);
}

// ─── WasmClassifier ──────────────────────────────────────────────

fn classifier() -> Option<WasmClassifier> {
    let bytes = model_bytes(CLASSIFY_KJQ)?;
    Some(WasmClassifier::load_core(&bytes).expect("load classifier"))
}

#[test]
fn classifier_reports_its_labels() {
    let Some(clf) = classifier() else {
        eprintln!("skipping: {CLASSIFY_KJQ} not present");
        return;
    };
    assert_eq!(clf.num_labels(), 2);
    assert_eq!(
        clf.labels(),
        vec!["NEGATIVE".to_string(), "POSITIVE".to_string()]
    );
}

#[test]
fn classifier_separates_positive_from_negative() {
    let Some(clf) = classifier() else { return };

    let positive = clf
        .classify_core("I love this, it is absolutely wonderful.")
        .unwrap();
    let negative = clf
        .classify_core("This is terrible and I want my money back.")
        .unwrap();

    assert_eq!(positive[0].label, "POSITIVE", "got {positive:?}");
    assert_eq!(negative[0].label, "NEGATIVE", "got {negative:?}");
    assert!(
        positive[0].score > 0.9,
        "weak confidence: {:?}",
        positive[0]
    );
    assert!(
        negative[0].score > 0.9,
        "weak confidence: {:?}",
        negative[0]
    );
}

#[test]
fn classifier_returns_every_label_highest_first() {
    let Some(clf) = classifier() else { return };
    let results = clf.classify_core("The film was fine.").unwrap();

    assert_eq!(results.len(), 2, "both labels should be present");
    assert!(
        results[0].score >= results[1].score,
        "results are not sorted: {results:?}"
    );

    let total: f32 = results.iter().map(|r| r.score).sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "softmax scores should sum to 1, got {total}"
    );
}

#[test]
fn classifier_index_points_into_labels() {
    // The demo will map index back onto labels(), so they have to agree.
    let Some(clf) = classifier() else { return };
    let labels = clf.labels();

    for r in clf.classify_core("Wonderful experience.").unwrap() {
        assert_eq!(
            labels[r.index], r.label,
            "index {} does not name {}",
            r.index, r.label
        );
    }
}

#[test]
fn classifier_batch_returns_one_result_per_input_in_order() {
    let Some(clf) = classifier() else { return };

    let texts = vec![
        "Absolutely fantastic, exceeded expectations.".to_string(),
        "Worst purchase of my life.".to_string(),
        "I am delighted with it.".to_string(),
    ];
    let out = clf.classify_batch_core(&texts).unwrap();

    assert_eq!(out.len(), texts.len(), "one result per input");
    assert_eq!(out[0].label, "POSITIVE");
    assert_eq!(out[1].label, "NEGATIVE");
    assert_eq!(out[2].label, "POSITIVE");
}

#[test]
fn classifier_batch_matches_classifying_one_at_a_time() {
    // Batching must not change the answer.
    let Some(clf) = classifier() else { return };

    let texts = vec![
        "A genuinely great product.".to_string(),
        "Completely useless.".to_string(),
    ];
    let batched = clf.classify_batch_core(&texts).unwrap();

    for (i, text) in texts.iter().enumerate() {
        let alone = clf.classify_core(text).unwrap();
        assert_eq!(
            batched[i].label, alone[0].label,
            "batch disagrees on {text:?}"
        );
        assert!(
            (batched[i].score - alone[0].score).abs() < 1e-4,
            "batch score drifted on {text:?}"
        );
    }
}

#[test]
fn classifier_rejects_an_embedding_model() {
    // A sentence encoder has no classification head; loading one must fail rather
    // than silently produce meaningless labels.
    let Some(bytes) = model_bytes(EMBED_KJQ) else {
        return;
    };
    assert!(
        WasmClassifier::load_core(&bytes).is_err(),
        "an embedding model must not load as a classifier"
    );
}

// ─── WasmChat ────────────────────────────────────────────────────

const CHAT_KJQ: &str = "../../../web.kjarni.ai/src/static/models/qwen05b-q8.kjq";

/// The chat model, loaded once for the whole binary.
///
/// Each test used to load its own. That means reading 500MB from disk and
/// dequantising half a billion int8 weights to f32 per test, several times over in
/// parallel, which is a few gigabytes of resident memory and most of the runtime.
/// Generation is not re-entrant either, so the mutex is doing real work rather than
/// just satisfying `Sync`.
fn chat() -> Option<std::sync::MutexGuard<'static, kjarni_wasm::WasmChat>> {
    use std::sync::{Mutex, OnceLock};
    static CHAT: OnceLock<Option<Mutex<kjarni_wasm::WasmChat>>> = OnceLock::new();

    CHAT.get_or_init(|| {
        let bytes = model_bytes(CHAT_KJQ)?;
        Some(Mutex::new(
            kjarni_wasm::WasmChat::load_core(&bytes, Some("qwen2.5-0.5b-instruct"))
                .expect("load chat"),
        ))
    })
    .as_ref()
    .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
}

#[test]
fn chat_loads_and_reports_its_context_window() {
    let Some(chat) = chat() else {
        eprintln!("skipping: {CHAT_KJQ} not present");
        return;
    };
    assert_eq!(chat.context_size(), 32768, "Qwen2.5 has a 32K window");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
fn chat_generates_a_factual_completion() {
    // Greedy, so this is deterministic. A weaker assertion than it looks: what it
    // really catches is the decode loop producing garbage, which is the failure mode
    // a broken KV cache or sampler actually gives.
    let Some(chat) = chat() else { return };

    let out = chat
        .generate_core("The capital of Iceland is", 16, 0.0)
        .unwrap();
    assert!(
        out.to_lowercase().contains("reykjav"),
        "expected the model to name Reykjavik, got {out:?}"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
fn chat_is_deterministic_at_temperature_zero() {
    let Some(chat) = chat() else { return };

    let a = chat.generate_core("Count: one, two,", 12, 0.0).unwrap();
    let b = chat.generate_core("Count: one, two,", 12, 0.0).unwrap();
    assert_eq!(a, b, "greedy decoding should not vary between calls");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
fn chat_respects_max_new_tokens() {
    let Some(chat) = chat() else { return };

    let short = chat
        .generate_core("Tell me about Iceland.", 8, 0.0)
        .unwrap();
    let long = chat
        .generate_core("Tell me about Iceland.", 48, 0.0)
        .unwrap();
    assert!(
        long.len() > short.len(),
        "a larger budget should produce more text: {} vs {}",
        short.len(),
        long.len()
    );
}

#[test]
fn chat_rejects_an_encoder_model() {
    // MiniLM has no decoder; loading it as a chat model must fail rather than
    // produce nonsense.
    let Some(bytes) = model_bytes(EMBED_KJQ) else {
        return;
    };
    assert!(kjarni_wasm::WasmChat::load_core(&bytes, None).is_err());
}

/// Generation must hand tokens out as they are decoded, not in one lump.
///
/// It used to do the latter: `generate_core` blocked on the whole loop and only
/// then drained the channel, so a caller could not show anything until the reply
/// was finished. The engine had been streaming into that channel the entire time.
///
/// Counting callbacks is what pins the fix. Asserting only on the returned text
/// would pass just as happily against the batched version.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
fn chat_streams_tokens_as_they_are_generated() {
    let Some(chat) = chat() else {
        eprintln!("skipping: {CHAT_KJQ} not present");
        return;
    };

    let mut pieces: Vec<String> = Vec::new();
    let text = chat
        .generate_stream_core("Count: one, two,", 12, 0.0, |piece| {
            pieces.push(piece.to_string())
        })
        .expect("generation succeeds");

    assert!(
        pieces.len() > 1,
        "expected several streamed pieces, got {}: {pieces:?}",
        pieces.len()
    );
    assert_eq!(
        pieces.concat(),
        text,
        "the streamed pieces must reconstruct exactly what was returned"
    );
}

/// Streaming and non-streaming must produce identical text.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
fn streaming_and_batched_generation_agree() {
    let Some(chat) = chat() else { return };

    let batched = chat
        .generate_core("The capital of Iceland is", 12, 0.0)
        .unwrap();
    let streamed = chat
        .generate_stream_core("The capital of Iceland is", 12, 0.0, |_| {})
        .unwrap();

    assert_eq!(
        batched, streamed,
        "greedy decoding must not vary by call path"
    );
}

/// Prefix reuse in the browser path.
///
/// `WasmChat` calls `run_generation_loop` directly rather than going through
/// `DecoderGenerator::stream`, because wasm has no blocking pool for
/// `spawn_blocking`, so the native wiring did not reach it and this needed its own.
/// A tab is one conversation, which is the shape the cache serves.
///
/// Loads its own instance rather than sharing `chat()`: enabling the cache needs
/// `&mut`, and it would otherwise leak into every other test in this binary.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
fn chat_prefix_cache_is_off_by_default_and_does_not_change_output() {
    let Some(bytes) = model_bytes(CHAT_KJQ) else {
        eprintln!("skipping: {CHAT_KJQ} not present");
        return;
    };

    // A long shared head, so the second call has something worth skipping.
    let shared = "Reference: Paris is the capital of France. Berlin is the capital \
                  of Germany. Rome is the capital of Italy. "
        .repeat(8);
    let prompt = format!("{shared}\nThe capital of Iceland is");

    let plain = kjarni_wasm::WasmChat::load_core(&bytes, Some("qwen2.5-0.5b-instruct"))
        .expect("load plain chat");
    assert!(
        !plain.has_prefix_cache(),
        "prefix reuse must be off until asked for"
    );
    let expected = plain.generate_core(&prompt, 16, 0.0).unwrap();

    let mut cached = kjarni_wasm::WasmChat::load_core(&bytes, Some("qwen2.5-0.5b-instruct"))
        .expect("load cached chat");
    cached.enable_prefix_cache(0).expect("enable prefix cache");
    assert!(cached.has_prefix_cache(), "prefix reuse should be on");

    let first = cached.generate_core(&prompt, 16, 0.0).unwrap();
    assert_eq!(
        expected, first,
        "a cold prefix cache changed the browser path's output"
    );

    // Same head, different tail: this is the call that should reuse.
    let second_prompt = format!("{shared}\nThe capital of Norway is");
    let reference = plain.generate_core(&second_prompt, 16, 0.0).unwrap();
    let warm = cached.generate_core(&second_prompt, 16, 0.0).unwrap();
    assert_eq!(
        reference, warm,
        "reusing a prefix changed the browser path's answer"
    );
}
