//! Compares raw logits after the prompt, which text comparison cannot do.
//!
//! Generated text tells you two runs diverged, not why. A near-tie between the
//! top two logits is precision and expected; a large gap is a real difference in
//! attention, RoPE or the LM head. And by the second token the two histories
//! differ, so anything after the prefill compares runs that already disagree.
//!
//! Reference from `bench/qwen_logits.py`. Both sides f32.
//!
//!     cargo test --release -p kjarni-models --test qwen_logits_parity -- --nocapture

use std::path::PathBuf;

use kjarni_models::models::qwen::QwenModel;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::pipeline::DecoderLoader;
use kjarni_transformers::tensor::DType;
use kjarni_transformers::traits::Device;
use ndarray::{Array2, Array3};

fn model_dir() -> Option<PathBuf> {
    let d = kjarni_transformers::models::get_default_cache_dir().join("Qwen_Qwen2.5-0.5B-Instruct");
    d.join("model.safetensors").exists().then_some(d)
}

fn reference() -> Option<serde_json::Value> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/torch_logits.json");
    serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()
}

#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "run with --release")]
async fn qwen_prompt_logits_match_pytorch() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: qwen2.5-0.5b not cached");
        return;
    };
    let Some(reference) = reference() else {
        eprintln!("skipping: bench/torch_logits.json not present");
        eprintln!("  regenerate with: cd bench && .venv/bin/python qwen_logits.py");
        return;
    };

    let cfg = ModelLoadConfig {
        target_dtype: Some(DType::F32),
        ..Default::default()
    };
    let model: QwenModel =
        DecoderLoader::load_from_pretrained(&dir, Device::Cpu, None, Some(cfg), None)
            .expect("load qwen f32");

    // The exact ids torch used, taken from the reference rather than
    // re-tokenised, so a tokenizer difference cannot masquerade as a model one.
    let ids: Vec<u32> = reference["prompt_token_ids"]
        .as_array()
        .expect("prompt_token_ids")
        .iter()
        .map(|v| v.as_u64().expect("token id") as u32)
        .collect();
    let seq = ids.len();

    use kjarni_transformers::decoder::traits::DecoderLanguageModel;
    let ops = model
        .decoder_cpu_ops()
        .expect("this model must have a CPU decoder");

    let tokens = Array2::from_shape_vec((1, seq), ids.clone()).expect("tokens");
    let mask = Array2::<f32>::ones((1, seq));

    let logits: Array3<f32> = ops
        .forward_to_logits(&tokens, &mask, 0, None)
        .expect("forward_to_logits");

    // Only the last position matters: that is what the next token is drawn from.
    let last = logits.slice(ndarray::s![0, seq - 1, ..]).to_owned();

    let mut ranked: Vec<(usize, f32)> = last.iter().cloned().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\n  prompt ids: {ids:?}");
    eprintln!("       torch    kjarni       delta  token");

    let empty = vec![];
    let top = reference["top10"].as_array().unwrap_or(&empty);
    let mut worst: f32 = 0.0;
    for entry in top.iter().take(5) {
        let id = entry["id"].as_u64().expect("id") as usize;
        let t = entry["logit"].as_f64().expect("logit") as f32;
        let k = last[id];
        worst = worst.max((t - k).abs());
        eprintln!(
            "  {t:>10.4}  {k:>8.4}  {:>10.4}  {}",
            t - k,
            entry["piece"].as_str().unwrap_or("")
        );
    }

    let torch_top = top[0]["id"].as_u64().expect("id") as usize;
    eprintln!("\n  torch argmax : {torch_top}");
    eprintln!("  kjarni argmax: {}", ranked[0].0);
    eprintln!("  gap to second: {:.4}", ranked[0].1 - ranked[1].1);
    eprintln!("  worst |delta| over the top 5: {worst:.4}\n");

    assert_eq!(
        ranked[0].0, torch_top,
        "the first predicted token must match; the prompt is identical so this is the model"
    );
    // f32 through 24 layers accumulates a little; a real bug shows up far larger.
    assert!(
        worst < 0.05,
        "logits drifted by {worst:.4}, which is beyond f32 accumulation"
    );
}
