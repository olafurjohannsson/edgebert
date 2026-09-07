//! Does Kjarni generate what PyTorch generates, given the same configuration?
//!
//! Getting this comparison right took three corrections, each of which silently
//! compared two different computations:
//!
//!   1. Qwen ships bf16 weights. Loading with `target_dtype: None` means "detect
//!      from file", so the engine ran bf16 while torch had been forced to f32.
//!   2. `generation_config.json` sets `repetition_penalty: 1.1`, and transformers
//!      applies it even under `do_sample=False`, because a repetition penalty is
//!      a logits processor rather than a sampling setting. Kjarni defaults to 1.0.
//!   3. Even matched to bf16 the two still differ, and correctly so: Kjarni keeps
//!      bf16 weights with f32 activations, while torch's bfloat16 mode makes the
//!      activations bf16 as well. Kjarni is the more precise of the two, and exact
//!      token parity between them is unreachable by construction.
//!
//! So parity is only meaningful with both pinned to f32 and a neutral penalty,
//! which is what this asserts.
//!
//! The reference comes from `bench/qwen_f32_parity.py`. Regenerate it if the
//! prompt or token budget changes here.
//!
//!     cargo test --release -p kjarni-models --test qwen_f32_parity -- --nocapture

use std::path::PathBuf;

use kjarni_models::models::qwen::QwenModel;
use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};
use kjarni_transformers::decoder::generator::DecoderGenerator;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::pipeline::DecoderLoader;
use kjarni_transformers::tensor::DType;
use kjarni_transformers::traits::Device;

const PROMPT: &str = "The capital of Iceland is";
const MAX_NEW_TOKENS: usize = 16;

fn model_dir() -> Option<PathBuf> {
    let d = kjarni_transformers::models::get_default_cache_dir().join("Qwen_Qwen2.5-0.5B-Instruct");
    d.join("model.safetensors").exists().then_some(d)
}

/// The text PyTorch produces for the same prompt, f32, penalty 1.0.
fn torch_reference() -> Option<String> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/torch_f32_parity.json");
    let raw = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    Some(v.get("text")?.as_str()?.to_string())
}

#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "generation is far too slow unoptimised")]
async fn qwen_f32_matches_pytorch() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: qwen2.5-0.5b not cached");
        return;
    };
    let Some(reference) = torch_reference() else {
        eprintln!("skipping: bench/torch_f32_parity.json not present");
        eprintln!("  regenerate with: cd bench && .venv/bin/python qwen_f32_parity.py");
        return;
    };

    // f32 explicitly. `None` would mean "detect from file", which is bf16 here.
    let cfg = ModelLoadConfig {
        target_dtype: Some(DType::F32),
        ..Default::default()
    };
    let model: QwenModel =
        DecoderLoader::load_from_pretrained(&dir, Device::Cpu, None, Some(cfg), None)
            .expect("load qwen as f32");

    let generator = DecoderGenerator::new(std::sync::Arc::new(model)).expect("generator");
    // Qwen's tokenizer adds no BOS: its prompt is exactly the 5 tokens
    // [785, 6722, 315, 38396, 374]. `GenerationConfig` defaults `add_bos_token`
    // to true, which prepends one the model was not trained to expect there and
    // changes the very first logits.
    let gen_cfg = GenerationConfig {
        max_new_tokens: Some(MAX_NEW_TOKENS),
        repetition_penalty: 1.0,
        add_bos_token: false,
        // Explicitly greedy. `GenerationConfig::default()` is
        // `Sample { temperature: 0.7 }`, so leaving this out does not compare
        // against PyTorch's greedy run at all: it samples, and three runs of this
        // test gave three different completions before the strategy was pinned.
        strategy: DecodingStrategy::Greedy,
        ..Default::default()
    };

    let got = generator
        .generate(PROMPT, &gen_cfg, None)
        .await
        .expect("generate");

    eprintln!("\n  prompt: {PROMPT:?}");
    eprintln!("  torch : {:?}", reference.trim());
    eprintln!("  kjarni: {:?}", got.trim());

    // Compare on the shorter of the two: the reference is a fixed 16 tokens, and
    // Kjarni may stop earlier at an EOS.
    let (a, b) = (reference.trim(), got.trim());
    let n = a.len().min(b.len());
    if a[..n] == b[..n] {
        eprintln!("  match over {n} characters\n");
    } else {
        let diverge = a[..n]
            .char_indices()
            .zip(b[..n].chars())
            .find(|((_, x), y)| x != y)
            .map(|((i, _), _)| i)
            .unwrap_or(n);
        eprintln!("  diverges at character {diverge}\n");
        panic!("f32 greedy output must match PyTorch exactly.\n  torch : {a:?}\n  kjarni: {b:?}");
    }
}
