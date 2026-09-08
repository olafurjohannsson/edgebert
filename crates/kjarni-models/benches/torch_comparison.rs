//! The Kjarni half of the PyTorch comparison.
//!
//! Mirrors `bench/torch_side.py` exactly: same models, same batch sizes, same
//! prompt, best of N after one warm-up, threads pinned. Emits JSON on stdout so
//! the two halves can be diffed rather than eyeballed.
//!
//!     RAYON_NUM_THREADS=8 cargo bench -p kjarni-models --bench torch_comparison
//!     RAYON_NUM_THREADS=8 .venv/bin/python bench/torch_side.py
//!
//! A bench rather than a test, deliberately. It asserts nothing: it measures, and
//! a measurement that fails a threshold on a busy machine is a flaky test rather
//! than a finding. Nothing gates on it, and it never runs in a normal pass.
//!
//! Compare with min-of-N over interleaved runs. A single run of each side is
//! noise, and the two halves must be run under the same thread count or the
//! comparison is partly measuring the scheduler.

use std::time::Instant;

use kjarni_models::SentenceEncoder;
use kjarni_transformers::traits::Device;

/// Best of `runs` after one warm-up.
///
/// Async rather than wrapping `block_on`: this runs inside a runtime already, and
/// blocking on a runtime from a thread that runtime is driving panics.
async fn timed<F, Fut, T>(runs: usize, mut f: F) -> (f64, Vec<f64>)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    f().await; // the first call pays for pages the timed runs should not
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let _ = f().await;
        times.push(t.elapsed().as_secs_f64());
    }
    (times.iter().cloned().fold(f64::MAX, f64::min), times)
}

fn ms(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{:.3}", x * 1000.0)).collect();
    format!("[{}]", parts.join(", "))
}

fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(bench_against_torch());
}

async fn bench_against_torch() {
    let runs: usize = std::env::var("BENCH_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let tokens: usize = std::env::var("BENCH_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let threads = rayon::current_num_threads();

    let cache = kjarni_transformers::models::get_default_cache_dir();
    let minilm = cache.join("sentence-transformers_all-MiniLM-L6-v2");
    let qwen = cache.join("Qwen_Qwen2.5-0.5B-Instruct");

    println!("{{");
    println!("  \"threads\": {threads},");

    // ── MiniLM ────────────────────────────────────────────────────
    let t = Instant::now();
    let encoder = SentenceEncoder::from_pretrained(&minilm, Device::Cpu, None, None, None)
        .expect("load minilm");
    println!("  \"minilm_load_s\": {:.4},", t.elapsed().as_secs_f64());

    let corpus: Vec<String> = (0..64)
        .map(|i| format!("Document number {i} about refunds, delivery and account settings."))
        .collect();

    println!("  \"minilm\": {{");
    for (idx, n) in [1usize, 8, 64].iter().enumerate() {
        let batch: Vec<&str> = corpus[..*n].iter().map(|s| s.as_str()).collect();
        let (best, all) = timed(runs, || async {
            encoder.encode_batch(&batch).await.expect("encode")
        })
        .await;
        let comma = if idx == 2 { "" } else { "," };
        println!(
            "    \"{n}\": {{\"best_ms\": {:.3}, \"all_ms\": {}}}{comma}",
            best * 1000.0,
            ms(&all)
        );
    }
    println!("  }},");

    // The same fixed sentence torch_side.py reports, so the two can be checked
    // against each other before any timing is believed.
    let reference = encoder
        .encode("The capital of Iceland is Reykjavik.")
        .await
        .expect("encode reference");
    let head: Vec<String> = reference[..16].iter().map(|v| format!("{v:.6}")).collect();
    println!("  \"minilm_dim\": {},", reference.len());
    println!("  \"minilm_reference_vector\": [{}],", head.join(", "));

    // ── Qwen ──────────────────────────────────────────────────────
    use kjarni_transformers::common::GenerationConfig;
    use kjarni_transformers::decoder::generator::DecoderGenerator;
    use kjarni_transformers::pipeline::DecoderLoader;

    let t = Instant::now();
    let model: kjarni_models::models::qwen::QwenModel =
        DecoderLoader::load_from_pretrained(&qwen, Device::Cpu, None, None, None)
            .expect("load qwen");
    let generator = DecoderGenerator::new(std::sync::Arc::new(model)).expect("generator");
    println!("  \"qwen_load_s\": {:.4},", t.elapsed().as_secs_f64());

    let config = GenerationConfig {
        max_new_tokens: Some(tokens),
        ..Default::default()
    };
    let (best, all) = timed(runs.max(2) / 2, || async {
        generator
            .generate("The capital of Iceland is", &config, None)
            .await
            .expect("generate")
    })
    .await;
    let text = generator
        .generate("The capital of Iceland is", &config, None)
        .await
        .expect("generate");

    println!("  \"qwen\": {{");
    println!("    \"tokens\": {tokens},");
    println!("    \"best_ms\": {:.3},", best * 1000.0);
    println!("    \"all_ms\": {},", ms(&all));
    println!("    \"tokens_per_s\": {:.3},", tokens as f64 / best);
    println!("    \"text\": {:?}", text.trim());
    println!("  }}");
    println!("}}");
}
