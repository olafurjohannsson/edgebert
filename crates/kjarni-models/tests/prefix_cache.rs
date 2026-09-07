//! Prefix reuse across generations.
//!
//! Every generation used to build a fresh KV cache and drop it, so a conversation
//! re-processed its whole history each turn. These check that carrying the cache
//! gives the same answer and skips the work.

use kjarni_models::models::qwen::QwenModel;
use kjarni_transformers::cache::PrefixCache;
use kjarni_transformers::common::stream::TokenType;
use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};
use kjarni_transformers::decoder::backend::AnyDecoderBackend;
use kjarni_transformers::decoder::generator::run_generation_loop_with_cache;
use kjarni_transformers::models::registry::ModelType;
use kjarni_transformers::{Device, LanguageModel};
use std::sync::Arc;
use std::time::Instant;

async fn load() -> Arc<QwenModel> {
    Arc::new(
        QwenModel::from_registry(
            ModelType::Qwen2_5_0_5B_Instruct,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await
        .expect("load qwen2.5-0.5b"),
    )
}

async fn generate(
    model: Arc<QwenModel>,
    tokens: Vec<u32>,
    prefix: Option<&mut PrefixCache>,
) -> (String, f64) {
    let backend = AnyDecoderBackend::cpu();
    // Greedy, not the default sampler: these tests compare two runs for equality,
    // and temperature 0.7 makes identical inputs produce different text.
    let cfg = GenerationConfig {
        max_new_tokens: Some(12),
        strategy: DecodingStrategy::Greedy,
        ..Default::default()
    };
    let (tx, mut rx) = tokio::sync::mpsc::channel(256);

    let t0 = Instant::now();
    let m: Arc<dyn kjarni_transformers::decoder::traits::DecoderLanguageModel + Send + Sync> =
        model.clone();
    let run = run_generation_loop_with_cache(m, backend, tokens, cfg, tx, None, prefix);

    let collect = async {
        let mut out = String::new();
        while let Some(Ok(t)) = rx.recv().await {
            if t.token_type != TokenType::Prompt {
                out.push_str(&t.text);
            }
        }
        out
    };
    let (res, text) = tokio::join!(run, collect);
    res.expect("generation");
    (text, t0.elapsed().as_secs_f64())
}

#[tokio::test]
#[ignore = "loads a real model from the local cache"]
async fn reused_prefix_gives_the_same_text() {
    let model = load().await;
    let tok = model.tokenizer();

    let base = "The capital of France is Paris. The capital of Germany is Berlin. \
                The capital of Italy is Rome. The capital of Spain is";
    let ids: Vec<u32> = tok.encode(base, false).unwrap().get_ids().to_vec();

    let (cold, t_cold) = generate(model.clone(), ids.clone(), None).await;

    let mut pc = PrefixCache::new(model.new_cache(1, 2048, 1).expect("cache"), 2048);
    let (warm_first, _) = generate(model.clone(), ids.clone(), Some(&mut pc)).await;
    let (warm_again, t_warm) = generate(model.clone(), ids.clone(), Some(&mut pc)).await;

    println!("\n  cold        {t_cold:>6.2}s  {cold:?}");
    println!("  warm again  {t_warm:>6.2}s  {warm_again:?}");
    println!(
        "  reused {} of {} tokens",
        pc.len().min(ids.len()),
        ids.len()
    );

    assert_eq!(cold, warm_first, "first warm run must match the cold one");
    assert_eq!(
        cold, warm_again,
        "reusing the prefix must not change the text"
    );
}

#[tokio::test]
#[ignore = "loads a real model from the local cache"]
async fn a_diverging_prompt_falls_back_correctly() {
    let model = load().await;
    let tok = model.tokenizer();

    let a = "The capital of France is Paris. The capital of Germany is";
    let b = "The capital of France is Paris. The capital of Japan is";
    let ids_a: Vec<u32> = tok.encode(a, false).unwrap().get_ids().to_vec();
    let ids_b: Vec<u32> = tok.encode(b, false).unwrap().get_ids().to_vec();

    let (want_b, _) = generate(model.clone(), ids_b.clone(), None).await;

    // Warm on A, then ask B: the shared prefix is reused and the rest replaced.
    let mut pc = PrefixCache::new(model.new_cache(1, 2048, 1).expect("cache"), 2048);
    let _ = generate(model.clone(), ids_a, Some(&mut pc)).await;
    let (got_b, _) = generate(model.clone(), ids_b, Some(&mut pc)).await;

    println!("\n  fresh B  {want_b:?}");
    println!("  after A  {got_b:?}");
    assert_eq!(
        want_b, got_b,
        "a cache warmed on a different prompt must not leak into this one"
    );
}
