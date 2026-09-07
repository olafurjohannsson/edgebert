//! Does the speculative decoding path work?
//!
//! `run_speculative_generation_loop` has been in the tree with no caller, no test
//! and no way to reach it from the public API: nothing anywhere sets
//! `speculation: Some(..)`. Qwen2.5-0.5B and 1.5B share a vocabulary (151,936),
//! which is what a draft model needs, so the pair can actually be run.

use kjarni_models::models::qwen::QwenModel;
use kjarni_transformers::common::stream::TokenType;
use kjarni_transformers::common::{DecodingStrategy, GenerationConfig, SpeculationParams};
use kjarni_transformers::cpu::decoder::{CpuDecoderBackend, run_speculative_generation_loop};
use kjarni_transformers::decoder::backend::AnyDecoderBackend;
use kjarni_transformers::decoder::generator::run_generation_loop;
use kjarni_transformers::decoder::traits::DecoderLanguageModel;
use kjarni_transformers::models::registry::ModelType;
use kjarni_transformers::{Device, LanguageModel};
use std::sync::Arc;
use std::time::Instant;

async fn load(mt: ModelType) -> Arc<QwenModel> {
    Arc::new(
        QwenModel::from_registry(mt, None, Device::Cpu, None, None)
            .await
            .expect("load"),
    )
}

fn greedy(n: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: Some(n),
        strategy: DecodingStrategy::Greedy,
        ..Default::default()
    }
}

/// Greedy with the repetition penalty the CLI actually defaults to.
///
/// Speculation ignored `repetition_penalty` entirely, so `--draft` silently
/// produced different, more repetitive text than the same command without it.
/// The default config has penalty 1.0, which hid the bug.
fn greedy_penalised(n: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: Some(n),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.1,
        ..Default::default()
    }
}

async fn collect(
    mut rx: tokio::sync::mpsc::Receiver<anyhow::Result<kjarni_transformers::common::StreamedToken>>,
) -> String {
    let mut out = String::new();
    while let Some(Ok(t)) = rx.recv().await {
        if t.token_type != TokenType::Prompt {
            out.push_str(&t.text);
        }
    }
    out
}

#[tokio::test]
#[ignore = "loads two real models from the local cache"]
async fn speculative_matches_plain_greedy() {
    let target = load(ModelType::Qwen2_5_1_5B_Instruct).await;
    let draft = load(ModelType::Qwen2_5_0_5B_Instruct).await;

    assert_eq!(
        target.vocab_size(),
        draft.vocab_size(),
        "a draft model must share the target's vocabulary"
    );

    let prompt = "The capital of France is Paris. The capital of Germany is";
    let ids: Vec<u32> = target
        .tokenizer()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec();

    // plain decoding, for both the answer and the time to beat
    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let t0 = Instant::now();
    let m: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let run = run_generation_loop(
        m,
        AnyDecoderBackend::cpu(),
        ids.clone(),
        greedy(24),
        tx,
        None,
    );
    let (r, plain) = tokio::join!(run, collect(rx));
    r.expect("plain generation");
    let t_plain = t0.elapsed().as_secs_f64();

    // speculative, greedy acceptance so it must reproduce the same text exactly
    let (tx2, rx2) = tokio::sync::mpsc::channel(256);
    let spec = SpeculationParams {
        num_tokens: 4,
        probabilistic: false,
    };
    let t1 = Instant::now();
    let tm: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let dm: Arc<dyn DecoderLanguageModel + Send + Sync> = draft.clone();
    let run2 = run_speculative_generation_loop(
        tm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        dm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        ids,
        greedy(24),
        &spec,
        tx2,
        None,
    );
    let (r2, spec_text) = tokio::join!(run2, collect(rx2));
    r2.expect("speculative generation");
    let t_spec = t1.elapsed().as_secs_f64();

    println!("\n  plain        {t_plain:>6.2}s  {plain:?}");
    println!("  speculative  {t_spec:>6.2}s  {spec_text:?}");
    println!("  speedup      {:.2}x", t_plain / t_spec);

    // A speculative round accepts a batch at once, so it can overshoot
    // max_new_tokens by up to num_tokens. Compare the shared prefix.
    let n = plain.len().min(spec_text.len());
    assert_eq!(
        &plain[..n],
        &spec_text[..n],
        "greedy speculative decoding must reproduce plain greedy output"
    );
    assert!(n > 0, "produced nothing");
}

#[tokio::test]
#[ignore = "loads two real models from the local cache"]
async fn speculation_depth_sweep() {
    let target = load(ModelType::Qwen2_5_1_5B_Instruct).await;
    let draft = load(ModelType::Qwen2_5_0_5B_Instruct).await;

    let prompt = "Write a short list of European capitals. The capital of France is";
    let ids: Vec<u32> = target
        .tokenizer()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec();

    let (tx, rx) = tokio::sync::mpsc::channel(512);
    let t0 = Instant::now();
    let m: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let run = run_generation_loop(
        m,
        AnyDecoderBackend::cpu(),
        ids.clone(),
        greedy(48),
        tx,
        None,
    );
    let (r, _) = tokio::join!(run, collect(rx));
    r.expect("plain");
    let t_plain = t0.elapsed().as_secs_f64();
    println!(
        "\n  plain 1.5B         {t_plain:>6.2}s   {:.1} tok/s",
        48.0 / t_plain
    );
    println!(
        "\n  {:>6}  {:>9}  {:>10}  {:>8}",
        "draft k", "time", "tok/s", "speedup"
    );

    for k in [2usize, 4, 6, 8] {
        let (tx2, rx2) = tokio::sync::mpsc::channel(512);
        let spec = SpeculationParams {
            num_tokens: k,
            probabilistic: false,
        };
        let t1 = Instant::now();
        let tm: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
        let dm: Arc<dyn DecoderLanguageModel + Send + Sync> = draft.clone();
        let run2 = run_speculative_generation_loop(
            tm,
            AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
            dm,
            AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
            ids.clone(),
            greedy(48),
            &spec,
            tx2,
            None,
        );
        let (r2, _text) = tokio::join!(run2, collect(rx2));
        r2.expect("speculative");
        let t = t1.elapsed().as_secs_f64();
        println!(
            "  {:>6}  {:>8.2}s  {:>9.1}  {:>7.2}x",
            k,
            t,
            48.0 / t,
            t_plain / t
        );
    }
}

#[tokio::test]
#[ignore = "loads two real models from the local cache"]
async fn speculation_respects_repetition_penalty() {
    let target = load(ModelType::Qwen2_5_1_5B_Instruct).await;
    let draft = load(ModelType::Qwen2_5_0_5B_Instruct).await;

    let prompt = "The capital of France is Paris. The capital of Germany is";
    let ids: Vec<u32> = target
        .tokenizer()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec();

    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let m: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let run = run_generation_loop(
        m,
        AnyDecoderBackend::cpu(),
        ids.clone(),
        greedy_penalised(24),
        tx,
        None,
    );
    let (r, plain) = tokio::join!(run, collect(rx));
    r.expect("plain");

    let (tx2, rx2) = tokio::sync::mpsc::channel(256);
    let spec = SpeculationParams {
        num_tokens: 4,
        probabilistic: false,
    };
    let tm: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let dm: Arc<dyn DecoderLanguageModel + Send + Sync> = draft.clone();
    let run2 = run_speculative_generation_loop(
        tm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        dm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        ids,
        greedy_penalised(24),
        &spec,
        tx2,
        None,
    );
    let (r2, spec_text) = tokio::join!(run2, collect(rx2));
    r2.expect("speculative");

    println!("\n  plain        {plain:?}");
    println!("  speculative  {spec_text:?}");

    let n = plain.len().min(spec_text.len());
    assert_eq!(
        &plain[..n],
        &spec_text[..n],
        "speculation must apply the same logit penalties as plain decoding"
    );
}

#[tokio::test]
#[ignore = "loads two real models from the local cache"]
async fn speculation_does_not_emit_stop_tokens() {
    let target = load(ModelType::Qwen2_5_1_5B_Instruct).await;
    let draft = load(ModelType::Qwen2_5_0_5B_Instruct).await;

    // A prompt short enough that the model finishes and hits its stop token
    // well inside the budget, so the stop path is actually exercised.
    let prompt = "<|im_start|>user\nWhat is the capital of Spain? Answer in one \
                  sentence.<|im_end|>\n<|im_start|>assistant\n";
    let ids: Vec<u32> = target
        .tokenizer()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec();

    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let spec = SpeculationParams {
        num_tokens: 6,
        probabilistic: false,
    };
    let tm: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let dm: Arc<dyn DecoderLanguageModel + Send + Sync> = draft.clone();
    let run = run_speculative_generation_loop(
        tm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        dm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        ids,
        greedy(40),
        &spec,
        tx,
        None,
    );
    let (r, text) = tokio::join!(run, collect(rx));
    r.expect("speculative");

    println!("\n  {text:?}");
    // The stop check ran after the send, so every speculative reply ended with
    // the terminator as literal text.
    for marker in ["<|im_end|>", "<|endoftext|>"] {
        assert!(
            !text.contains(marker),
            "a stop token reached the stream as text: {text:?}"
        );
    }
    assert!(!text.is_empty(), "generated nothing");
}
