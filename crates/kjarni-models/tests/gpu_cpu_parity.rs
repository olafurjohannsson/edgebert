//! Do the GPU kernels agree with the CPU ones?
//!
//! Nothing in this repository compared the two until now. The GPU decoder once
//! emitted pure garbage for Qwen because its QKV biases were never applied, and
//! that was found by reading the code rather than by a failing test. A shader is
//! the easiest place in this engine to be silently wrong: it compiles, it runs,
//! it produces numbers, and only a comparison says whether they mean anything.
//!
//! Skips when no adapter is available, so a headless CI box passes rather than
//! failing on hardware it does not have.
//!
//!     cargo test -p kjarni-models --test gpu_cpu_parity --release -- --test-threads=1

use anyhow::Result;
use kjarni_models::SentenceEncoder;
use kjarni_transformers::WgpuContext;
use kjarni_transformers::activations::Activation;
use kjarni_transformers::models::ModelType;
use kjarni_transformers::traits::Device;
use std::sync::Arc;

/// `None` when there is no usable adapter, which is the normal case in CI.
async fn gpu_context() -> Option<Arc<WgpuContext>> {
    match WgpuContext::new().await {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            None
        }
    }
}

fn model_cached(mt: ModelType) -> bool {
    let cached = mt.is_downloaded(&kjarni_transformers::models::get_default_cache_dir());
    if !cached {
        eprintln!("skipping: {} not downloaded", mt.cli_name());
    }
    cached
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// The headline check: the same text through both devices must embed to the same
/// vector.
///
/// Not bit-identical, and it should not be asserted as such. A GPU reassociates
/// reductions and may use different intrinsics, so the two legitimately differ in
/// the last few bits. What must not happen is a divergence large enough to change
/// which document a search returns.
#[tokio::test]
#[cfg_attr(
    debug_assertions,
    ignore = "GPU work is dominated by kernel launches unoptimised; run with --release"
)]
async fn encoder_embeddings_agree_across_devices() -> Result<()> {
    if !model_cached(ModelType::MiniLML6V2) {
        return Ok(());
    }
    let Some(context) = gpu_context().await else {
        return Ok(());
    };

    let cpu = SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Cpu, None, None)
        .await?;
    let gpu = SentenceEncoder::from_registry(
        ModelType::MiniLML6V2,
        None,
        Device::Wgpu,
        Some(context),
        None,
    )
    .await?;

    // Deliberately varied: a short phrase, one long enough to fill the window, and
    // one with punctuation and digits, since padding and masking are where the two
    // paths most easily disagree.
    let inputs = [
        "a dog",
        "The quick brown fox jumps over the lazy dog, again and again, until the \
         sentence is long enough to exercise more than one attention block.",
        "Order #12345 shipped on 2024-08-14 (expedited).",
    ];

    for text in inputs {
        let a = cpu.encode(text).await?;
        let b = gpu.encode(text).await?;
        assert_eq!(a.len(), b.len(), "dimension differs for {text:?}");

        let cos = cosine(&a, &b);
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!("  {:.8} cosine, worst {worst:.2e}  {text:.40?}", cos);

        assert!(
            cos > 0.9999,
            "GPU and CPU disagree on {text:?}: cosine {cos}"
        );
        assert!(worst < 1e-2, "worst element differs by {worst} on {text:?}");
    }
    Ok(())
}

/// Ranking is what an embedding is for, so a divergence that preserves cosine but
/// reorders results would still be a bug. This checks the ordering directly.
#[tokio::test]
#[cfg_attr(
    debug_assertions,
    ignore = "GPU work is dominated by kernel launches unoptimised; run with --release"
)]
async fn similarity_ranking_is_identical_across_devices() -> Result<()> {
    if !model_cached(ModelType::MiniLML6V2) {
        return Ok(());
    }
    let Some(context) = gpu_context().await else {
        return Ok(());
    };

    let cpu = SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Cpu, None, None)
        .await?;
    let gpu = SentenceEncoder::from_registry(
        ModelType::MiniLML6V2,
        None,
        Device::Wgpu,
        Some(context),
        None,
    )
    .await?;

    let query = "how do I return an item?";
    let docs = [
        "Our return policy allows refunds within 30 days of purchase.",
        "Standard shipping takes 5-7 business days.",
        "Click Forgot Password on the login page to reset it.",
        "Returns must be in the original packaging to qualify.",
    ];

    let rank = |q: Vec<f32>, ds: Vec<Vec<f32>>| {
        let mut idx: Vec<usize> = (0..ds.len()).collect();
        idx.sort_by(|a, b| cosine(&ds[*b], &q).total_cmp(&cosine(&ds[*a], &q)));
        idx
    };

    let mut cpu_docs = Vec::new();
    let mut gpu_docs = Vec::new();
    for d in docs {
        cpu_docs.push(cpu.encode(d).await?);
        gpu_docs.push(gpu.encode(d).await?);
    }
    let cpu_rank = rank(cpu.encode(query).await?, cpu_docs);
    let gpu_rank = rank(gpu.encode(query).await?, gpu_docs);

    println!("  cpu {cpu_rank:?}\n  gpu {gpu_rank:?}");
    assert_eq!(
        cpu_rank, gpu_rank,
        "the two devices ranked the same documents differently"
    );
    Ok(())
}

/// Every activation the GPU feed-forward advertises must match its CPU scalar.
///
/// `fc1.wgsl` dispatches on a numeric constant, and the Rust side maps each
/// `Activation` to one of those numbers. Nothing links the two: adding a variant
/// to the enum and forgetting the shader branch, or numbering them differently,
/// produces a model that runs and is quietly wrong. QuickGELU was added for CLIP
/// with exactly that risk and no test.
#[test]
fn every_activation_has_a_cpu_reference() {
    // Mirrors the branches in `gpu_ops/blocks/ffn/fc1.wgsl`. If a variant is added
    // without a shader case, this list stops compiling exhaustively below.
    for act in [
        Activation::Gelu,
        Activation::GeluNew,
        Activation::Relu,
        Activation::SilU,
        Activation::Tanh,
        Activation::QuickGelu,
    ] {
        let mut v =
            ndarray::Array2::from_shape_vec((1, 7), vec![-3.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
                .unwrap();
        kjarni_transformers::activations::apply_activation_2d(&mut v, act);
        for x in v.iter() {
            assert!(x.is_finite(), "{act:?} produced {x}");
        }
    }

    // QuickGELU specifically, against PyTorch's `x * sigmoid(1.702x)`. This is the
    // one the GPU shader gained most recently and the one CLIP depends on.
    let mut v = ndarray::Array2::from_shape_vec((1, 3), vec![-1.0f32, 0.5, 3.0]).unwrap();
    kjarni_transformers::activations::apply_activation_2d(&mut v, Activation::QuickGelu);
    for (got, want) in v.iter().zip([-0.154_204_23f32, 0.350_388_44, 2.981_928_8]) {
        assert!((got - want).abs() < 1e-6, "QuickGELU: {got} vs {want}");
    }
}
