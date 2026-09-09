//! Correctness of the packed Q4_K GPU kernels against the CPU implementation.
//!
//! The GPU kernels read Q4_K blocks exactly as the GGUF stores them, so what can go
//! wrong is the unpacking: the six-bit scale/min reassembly for sub-blocks 4-7, and
//! which nibble belongs to which half of a 64-weight group. Both fail silently while
//! still producing plausible numbers, so these compare against `matmul_2d_cpu_q4_k`
//! on real weights rather than on synthetic blocks.
//!
//! Non-square tensors are covered deliberately. A square weight matrix cannot detect
//! a transposed shape convention, and `attn_q` happens to be square.

use anyhow::Result;
use ndarray::Array2;

use super::*;
use crate::WgpuContext;
use crate::cpu::kernels::q_common::BlockQ4_K;
use crate::gpu::GpuTensor;
use crate::tests::common::read_gpu_tensor_to_vec;
use crate::weights::ModelWeights;

fn gguf_path() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
}

/// Runs both paths over `m` rows of one real Q4_K tensor. `None` when the model or
/// the tensor is unavailable, so the suite still runs on a machine without them.
async fn run_q4k(m: usize, tensor: &str) -> Result<Option<(Array2<f32>, Array2<f32>, Vec<usize>)>> {
    let path = gguf_path();
    if !path.exists() {
        return Ok(None);
    }
    let weights = ModelWeights::new(&path)?;
    if weights.tensor_dtype(tensor).ok() != Some(DType::Q4_K) {
        return Ok(None);
    }

    let context = WgpuContext::new().await?;
    let linear = GpuLinearLayer::new(&context);

    // The real loader path, not a hand-built view: the shape and dtype it derives
    // from the GGUF are part of what is under test.
    let gpu_w = GpuTensor::from_model_weights(&context, &weights, tensor, None, "q4k_weights")?;
    let shape = gpu_w.shape().to_vec();
    let (n, k) = (shape[0], shape[1]);

    // Via `get_typed_tensor`, not the raw bytes: GGUF stores the Q and K projections
    // row-permuted for RoPE, and the conversion undoes that. The GPU upload goes
    // through the same conversion, so comparing against raw blocks would compare
    // against a differently-ordered matrix and fail on exactly those two tensors.
    let blocks: Vec<BlockQ4_K> = match weights.get_typed_tensor(tensor)? {
        crate::tensor::CpuTensor::Q4_K(m) => m.blocks.clone(),
        other => anyhow::bail!("{tensor} converted to {:?}, expected Q4_K", other.dtype()),
    };

    let mut state = 0x9E3779B97F4A7C15u64;
    let mut input = Vec::with_capacity(m * k);
    for _ in 0..m * k {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        input.push(((state >> 40) as f32 / 8_388_608.0) - 1.0);
    }
    let input = Array2::from_shape_vec((m, k), input)?;

    let gpu_in = GpuTensor::from_ndarray(&context, &input)?;
    let gpu_out = GpuTensor::uninitialized(&context, vec![m, n], DType::F32, "q4k_out");

    let mut enc = context.device.create_command_encoder(&Default::default());
    linear.encode(&mut enc, &gpu_in, &gpu_w, &gpu_out);
    context.queue.submit(std::iter::once(enc.finish()));

    let (v, out_shape) = read_gpu_tensor_to_vec::<f32>(&gpu_out).await?;
    let gpu = Array2::from_shape_vec((out_shape[0], out_shape[1]), v)?;
    // The exact f32 reference, not `matmul_2d_cpu_q4_k`. That function now quantises
    // its activations to Q8_K for speed, which would put ~2% of int8 error between the
    // two sides and blunt this test into checking nothing tighter than "roughly right".
    // The shader keeps activations in f32, so it is compared against the kernel that
    // does the same, and the tolerance can stay at f32 rounding.
    let mut cpu = Array2::<f32>::zeros((m, n));
    for r in 0..m {
        let row = input.row(r).to_owned();
        unsafe {
            crate::cpu::kernels::x86::q4_k::matmul_vec_q4_k_avx2(
                cpu.row_mut(r).as_slice_mut().unwrap(),
                row.as_ptr(),
                &blocks,
                k,
            )
        }
    }
    Ok(Some((gpu, cpu, shape)))
}

fn report(label: &str, gpu: &Array2<f32>, cpu: &Array2<f32>, shape: &[usize]) {
    let mut worst = 0.0f32;
    let mut scale = 0.0f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        worst = worst.max((g - c).abs());
        scale = scale.max(c.abs());
    }
    println!("  {label} {shape:?}: max abs diff {worst:.3e}, |cpu| up to {scale:.3}");
    // Both sides dequantise the same blocks, so only f32 summation order differs.
    // Anything larger means the unpacking disagrees, not the arithmetic.
    assert!(
        worst < 1e-2 * scale.max(1.0),
        "{label}: GPU and CPU Q4_K disagree, max abs diff {worst:.3e}"
    );
}

#[tokio::test]
#[ignore = "GPU required"]
async fn q4k_gemv_matches_cpu() -> Result<()> {
    for t in [
        "blk.0.attn_q.weight",   // square, 3072x3072
        "blk.0.attn_k.weight",   // GQA, narrow output
        "blk.0.ffn_gate.weight", // wide output, 8192x3072
        "blk.1.ffn_down.weight", // wide input, K=8192
        "blk.3.ffn_down.weight",
        "blk.5.ffn_down.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
    ] {
        match run_q4k(1, t).await? {
            None => println!("  skipping {t}"),
            Some((gpu, cpu, shape)) => report(t, &gpu, &cpu, &shape),
        }
    }
    Ok(())
}

#[tokio::test]
#[ignore = "GPU required"]
async fn q4k_batched_matches_cpu() -> Result<()> {
    for t in ["blk.0.attn_q.weight", "blk.0.ffn_gate.weight"] {
        match run_q4k(8, t).await? {
            None => println!("  skipping {t}"),
            Some((gpu, cpu, shape)) => report(t, &gpu, &cpu, &shape),
        }
    }
    Ok(())
}

/// Packed Q4_K against the same tensor expanded to BF16.
///
/// The tests above compare the packed path against the CPU Q4_K matmul, and both
/// sides read the blocks with the same [n, k] convention, so a consistent
/// transpose would cancel out and pass. Expanding to BF16 goes through
/// `get_typed_tensor`, an independent route into the same weights, so comparing
/// the two catches a disagreement about layout that neither can catch alone.
#[tokio::test]
#[ignore = "GPU required"]
async fn q4k_packed_matches_expanded() -> Result<()> {
    let path = gguf_path();
    if !path.exists() {
        println!("Skipping: model not present");
        return Ok(());
    }
    let weights = ModelWeights::new(&path)?;
    let context = WgpuContext::new().await?;
    let linear = GpuLinearLayer::new(&context);

    for name in ["blk.0.attn_q.weight", "blk.0.ffn_gate.weight"] {
        if weights.tensor_dtype(name).ok() != Some(DType::Q4_K) {
            continue;
        }
        let packed = GpuTensor::from_model_weights(&context, &weights, name, None, "packed")?;
        let expanded =
            GpuTensor::from_model_weights(&context, &weights, name, Some(DType::BF16), "expanded")?;
        println!(
            "  {name}: packed shape {:?} {:?}, expanded shape {:?} {:?}",
            packed.shape(),
            packed.dtype(),
            expanded.shape(),
            expanded.dtype()
        );
        assert_eq!(packed.shape(), expanded.shape(), "{name}: shapes disagree");

        let (n, k) = (packed.shape()[0], packed.shape()[1]);
        let mut state = 0xDEADBEEFCAFEBABEu64;
        let mut v = Vec::with_capacity(k);
        for _ in 0..k {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.push(((state >> 40) as f32 / 8_388_608.0) - 1.0);
        }
        let input = Array2::from_shape_vec((1, k), v)?;
        let gpu_in = GpuTensor::from_ndarray(&context, &input)?;

        let run = |w: &GpuTensor| -> Result<Vec<f32>> {
            let out = GpuTensor::uninitialized(&context, vec![1, n], DType::F32, "out");
            let mut enc = context.device.create_command_encoder(&Default::default());
            linear.encode(&mut enc, &gpu_in, w, &out);
            context.queue.submit(std::iter::once(enc.finish()));
            Ok(pollster::block_on(read_gpu_tensor_to_vec::<f32>(&out))?.0)
        };
        let a = run(&packed)?;
        let b = run(&expanded)?;

        let mut worst = 0.0f32;
        let mut scale = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            worst = worst.max((x - y).abs());
            scale = scale.max(y.abs());
        }
        println!("    max abs diff {worst:.3e} against |expanded| up to {scale:.3}");
        // BF16 rounds the dequantised values, so a small gap is expected; a layout
        // disagreement would show up orders of magnitude larger than this.
        assert!(
            worst < 5e-2 * scale.max(1.0),
            "{name}: packed and expanded disagree, {worst:.3e}"
        );
    }
    Ok(())
}

/// Achieved bandwidth of a single GEMV, packed Q4_K against expanded BF16.
///
/// Decode reads every weight once per token, so a GEMV kernel's only real figure of
/// merit is how close it gets to the card's bandwidth. Timing one projection in
/// isolation separates that from scheduling, RoPE, cache traffic and CPU overhead.
#[tokio::test]
#[ignore = "GPU required"]
async fn q4k_gemv_bandwidth() -> Result<()> {
    let path = gguf_path();
    if !path.exists() {
        println!("Skipping: model not present");
        return Ok(());
    }
    let weights = ModelWeights::new(&path)?;
    let context = WgpuContext::new().await?;
    let linear = GpuLinearLayer::new(&context);

    for name in ["blk.0.ffn_gate.weight", "blk.0.attn_q.weight"] {
        if weights.tensor_dtype(name).ok() != Some(DType::Q4_K) {
            continue;
        }
        let packed = GpuTensor::from_model_weights(&context, &weights, name, None, "p")?;
        let expanded =
            GpuTensor::from_model_weights(&context, &weights, name, Some(DType::BF16), "e")?;
        let (n, k) = (packed.shape()[0], packed.shape()[1]);

        let input = Array2::<f32>::zeros((1, k));
        let gpu_in = GpuTensor::from_ndarray(&context, &input)?;
        let out = GpuTensor::uninitialized(&context, vec![1, n], DType::F32, "o");

        for (label, w, bytes) in [
            ("packed Q4_K", &packed, n * k * 144 / 256),
            ("expanded BF16", &expanded, n * k * 2),
        ] {
            // 50 dispatches per submit so launch overhead is amortised away.
            let reps = 50;
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let mut enc = context.device.create_command_encoder(&Default::default());
                for _ in 0..reps {
                    linear.encode(&mut enc, &gpu_in, w, &out);
                }
                context.queue.submit(std::iter::once(enc.finish()));
                context.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })?;
                best = best.min(t.elapsed().as_secs_f64());
            }
            let per_call = best / reps as f64;
            println!(
                "  {name} [{n}x{k}] {label:<14}: {:>7.3} ms  {:>6.1} GB/s  ({:.0}% of 288)",
                per_call * 1e3,
                bytes as f64 / per_call / 1e9,
                100.0 * (bytes as f64 / per_call / 1e9) / 288.0
            );
        }
    }
    Ok(())
}

/// Packed Q6_K on the GPU, against the CPU Q6_K matmul and against the same tensor
/// expanded to BF16.
///
/// Both comparisons matter. The CPU one checks the unpacking arithmetic; the
/// expanded one is an independent route into the same weights and is what would
/// catch a layout or reordering disagreement, which is exactly the bug the Q4_K
/// path shipped with until it was compared this way.
#[tokio::test]
#[ignore = "GPU required"]
async fn q6k_packed_matches_cpu_and_expanded() -> Result<()> {
    use crate::cpu::kernels::q_common::BlockQ6_K;

    let path = gguf_path();
    if !path.exists() {
        println!("Skipping: model not present");
        return Ok(());
    }
    let weights = ModelWeights::new(&path)?;
    let context = WgpuContext::new().await?;
    let linear = GpuLinearLayer::new(&context);

    for name in [
        "output.weight",         // the lm_head, streamed every token
        "blk.1.ffn_down.weight", // wide input, K=8192
        "blk.0.attn_v.weight",   // narrow output
    ] {
        if weights.tensor_dtype(name).ok() != Some(DType::Q6_K) {
            println!("  skipping {name}");
            continue;
        }
        let packed = GpuTensor::from_model_weights(&context, &weights, name, None, "p")?;
        assert_eq!(packed.dtype(), DType::Q6_K, "{name} did not stay packed");
        let expanded =
            GpuTensor::from_model_weights(&context, &weights, name, Some(DType::BF16), "e")?;
        assert_eq!(packed.shape(), expanded.shape(), "{name}: shapes disagree");

        let (n, k) = (packed.shape()[0], packed.shape()[1]);
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut v = Vec::with_capacity(k);
        for _ in 0..k {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.push(((state >> 40) as f32 / 8_388_608.0) - 1.0);
        }
        let input = Array2::from_shape_vec((1, k), v)?;
        let gpu_in = GpuTensor::from_ndarray(&context, &input)?;

        let run = |w: &GpuTensor| -> Result<Vec<f32>> {
            let out = GpuTensor::uninitialized(&context, vec![1, n], DType::F32, "o");
            let mut enc = context.device.create_command_encoder(&Default::default());
            linear.encode(&mut enc, &gpu_in, w, &out);
            context.queue.submit(std::iter::once(enc.finish()));
            Ok(pollster::block_on(read_gpu_tensor_to_vec::<f32>(&out))?.0)
        };
        let gpu = run(&packed)?;
        let exp = run(&expanded)?;

        let blocks: Vec<BlockQ6_K> = match weights.get_typed_tensor(name)? {
            crate::tensor::CpuTensor::Q6_K(m) => m.blocks.clone(),
            other => anyhow::bail!("{name} converted to {:?}", other.dtype()),
        };
        let cpu = crate::cpu::ops::matmul::matmul_2d_cpu_q6_k(&input.view(), &blocks);

        let mut d_cpu = 0.0f32;
        let mut d_exp = 0.0f32;
        let mut scale = 0.0f32;
        for i in 0..n {
            scale = scale.max(cpu[[0, i]].abs());
            d_cpu = d_cpu.max((gpu[i] - cpu[[0, i]]).abs());
            d_exp = d_exp.max((gpu[i] - exp[i]).abs());
        }
        println!(
            "  {name} [{n}x{k}]: vs cpu {d_cpu:.3e}, vs expanded {d_exp:.3e}, |cpu| up to {scale:.3}"
        );
        // The CPU Q6_K path squeezes the activations to int8 via `quantize_row_q8_k`
        // before its dot product, while this kernel keeps them in f32. So the GPU is
        // the more accurate of the two here and a few tenths of a percent apart is
        // the expected result, not summation noise.
        assert!(
            d_cpu < 2e-2 * scale.max(1.0),
            "{name}: GPU vs CPU {d_cpu:.3e}"
        );
        // Against BF16 the expansion rounds, so allow more, but a layout mismatch
        // would be orders of magnitude beyond this.
        assert!(
            d_exp < 1e-1 * scale.max(1.0),
            "{name}: GPU vs expanded {d_exp:.3e}"
        );
    }
    Ok(())
}

/// A row range of a fused tensor must upload the same rows the CPU factory slices.
///
/// Phi-3 fuses q, k and v into one `qkv_proj`, and the CPU and GPU paths reach it
/// through different code: `CpuLayerFactory::build_linear` on one side and
/// `GpuTensor::from_model_weights` on the other. Taking the wrong third on either side
/// gives a model that still emits fluent text, so the two are compared directly.
#[tokio::test]
#[ignore = "GPU required"]
async fn fused_row_range_uploads_the_same_rows_the_cpu_slices() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/kjarni/microsoft_Phi-3.5-mini-instruct");
    if !dir.exists() {
        println!("Skipping: Phi-3.5-mini not present");
        return Ok(());
    }
    let weights = ModelWeights::new(&dir)?;
    let fused = "model.layers.0.self_attn.qkv_proj.weight";
    let hidden = 3072usize;

    let full = match weights.get_typed_tensor(fused)? {
        crate::tensor::CpuTensor::BF16(a) => a.mapv(|v| v.to_f32()),
        crate::tensor::CpuTensor::F32(a) => a,
        other => anyhow::bail!("unexpected dtype {:?}", other.dtype()),
    }
    .into_dimensionality::<ndarray::Ix2>()?;
    assert_eq!(
        full.shape(),
        &[3 * hidden, hidden],
        "Phi-3 fuses q, k and v"
    );

    let context = WgpuContext::new().await?;

    // q, k and v in turn, so a range taken from the wrong third is caught. Both upload
    // dtypes are exercised: BF16 is what the decoder actually asks for, and F32 goes
    // through the conversion branch.
    for (name, start) in [("q", 0), ("k", hidden), ("v", 2 * hidden)] {
        let ranged = format!("{fused}[{}:{}]", start, start + hidden);

        // BF16, read back as raw bits since the buffer holds two bytes per value.
        let gpu = GpuTensor::from_model_weights(&context, &weights, &ranged, None, name)?;
        assert_eq!(gpu.shape(), &[hidden, hidden], "{name} shape");
        assert_eq!(gpu.dtype(), DType::BF16, "{name} should stay BF16");

        let (bits, _) = read_gpu_tensor_to_vec::<u16>(&gpu).await?;
        let mut worst = 0.0f32;
        for col in 0..hidden {
            let got = half::bf16::from_bits(bits[col]).to_f32();
            worst = worst.max((got - full[[start, col]]).abs());
        }
        println!(
            "  {name} bf16 rows {start}..{}: max abs diff {worst:.3e}",
            start + hidden
        );
        // The source is already BF16, so the slice is copied bit for bit.
        assert!(
            worst == 0.0,
            "{name} bf16 did not upload the rows it was asked for"
        );

        // F32, which converts on the way and should still be the same rows.
        let gpu_f32 =
            GpuTensor::from_model_weights(&context, &weights, &ranged, Some(DType::F32), name)?;
        assert_eq!(gpu_f32.dtype(), DType::F32);
        let (values, _) = read_gpu_tensor_to_vec::<f32>(&gpu_f32).await?;
        let mut worst_f32 = 0.0f32;
        for col in 0..hidden {
            worst_f32 = worst_f32.max((values[col] - full[[start, col]]).abs());
        }
        println!(
            "  {name} f32  rows {start}..{}: max abs diff {worst_f32:.3e}",
            start + hidden
        );
        assert!(
            worst_f32 < 1e-6,
            "{name} f32 did not upload the rows it was asked for"
        );
    }
    Ok(())
}

// ─── Q8_0 ────────────────────────────────────────────────────────
//
// Unlike Q4_K these need no GGUF: Q8_0 is one f16 scale and 32 int8 values, so a
// matrix can be quantised on the spot and the reference is just the dequantised
// blocks multiplied out in f32. That makes the comparison exact by construction --
// both sides are meant to compute `sum(input * d * qs)` over the same blocks -- and
// it means the test runs on any machine with a GPU.
//
// What can actually go wrong is the repack. The stored block is 34 bytes, which
// leaves `qs` two bytes out of word alignment, so the upload moves the scale into a
// word of its own. If that layout and the shader's indexing ever disagree, the
// output stays plausible while being wrong, which is what these catch.

/// Quantise a deterministic pseudo-random `[n, k]` matrix, multiply an `[m, k]`
/// input by it on both devices, and return `(gpu, cpu)`.
async fn run_q8(m: usize, n: usize, k: usize) -> Result<(Array2<f32>, Array2<f32>)> {
    use crate::linear_layer::LinearData;
    use crate::tensor::QuantizedMatrix;
    use std::sync::Arc;

    let mut state = 0x243F6A8885A308D3u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / 8_388_608.0) - 1.0
    };

    let weights = Array2::from_shape_fn((n, k), |_| next());
    let input = Array2::from_shape_fn((m, k), |_| next());

    let blocks = crate::cpu::kernels::quantize::quantize_matrix_q8_0(&weights)?;
    let matrix = QuantizedMatrix {
        blocks,
        shape: [n, k],
    };

    // The reference: the same blocks the GPU gets, dequantised and multiplied in
    // f32. Not the original `weights`, which would fold quantisation error into the
    // comparison and hide a layout bug behind it.
    let dequantized = matrix.dequantize()?;
    let cpu = input.dot(&dequantized.t());

    let context = WgpuContext::new().await?;
    let linear = GpuLinearLayer::new(&context);

    // Through `LinearLayer::to_gpu_tensor`, because that is the path a loaded
    // decoder takes; a hand-built `TensorView` would skip the repack under test.
    let layer = crate::linear_layer::LinearLayer {
        data: LinearData::Q8_0(Arc::new(matrix)),
        bias: None,
        // Irrelevant here: the Q8_0 path never reaches the f32 dispatch.
        f32_strategy: crate::linear_layer::F32MatmulStrategy::Faer,
    };
    let gpu_w = layer.to_gpu_tensor(&context, "q8_weights")?;
    let gpu_in = GpuTensor::from_ndarray(&context, &input)?;
    let gpu_out = GpuTensor::uninitialized(&context, vec![m, n], DType::F32, "q8_out");

    let mut enc = context.device.create_command_encoder(&Default::default());
    linear.encode(&mut enc, &gpu_in, &gpu_w, &gpu_out);
    context.queue.submit(std::iter::once(enc.finish()));

    let (v, out_shape) = read_gpu_tensor_to_vec::<f32>(&gpu_out).await?;
    let gpu = Array2::from_shape_vec((out_shape[0], out_shape[1]), v)?;
    Ok((gpu, cpu))
}

#[tokio::test]
#[ignore = "GPU required"]
async fn q8_gemv_matches_cpu() -> Result<()> {
    // `n` deliberately includes values that are not multiples of the four rows a
    // GEMV workgroup owns, so the tail path is covered.
    for (n, k) in [(64, 64), (128, 256), (255, 128), (33, 512), (1, 32)] {
        let (gpu, cpu) = run_q8(1, n, k).await?;
        report(&format!("q8 gemv [{n}x{k}]"), &gpu, &cpu, &[n, k]);
    }
    Ok(())
}

#[tokio::test]
#[ignore = "GPU required"]
async fn q8_batched_matches_cpu() -> Result<()> {
    for (m, n, k) in [(4, 64, 128), (8, 129, 256), (2, 32, 64)] {
        let (gpu, cpu) = run_q8(m, n, k).await?;
        report(&format!("q8 bmm [{m}x{k} @ {n}]"), &gpu, &cpu, &[n, k]);
    }
    Ok(())
}

#[tokio::test]
#[ignore = "GPU required"]
async fn q8_repack_preserves_every_value() -> Result<()> {
    use crate::cpu::kernels::q_common::BlockQ8_0;
    use half::f16;

    // The repack is the one place a silent corruption could live, so check it
    // directly rather than only through a matmul that averages 128 values together.
    let blocks: Vec<BlockQ8_0> = (0..3)
        .map(|b| BlockQ8_0 {
            d: f16::from_f32(0.5 + b as f32),
            qs: std::array::from_fn(|i| (i as i32 - 16) as i8),
        })
        .collect();

    let packed = crate::gpu::tensor::repack_q8_0(&blocks);
    assert_eq!(packed.len(), blocks.len() * 36, "9 words per block");

    for (b, block) in blocks.iter().enumerate() {
        let o = b * 36;
        let d = f16::from_bits(u16::from_le_bytes([packed[o], packed[o + 1]]));
        assert_eq!(d, block.d, "block {b} scale");
        assert_eq!(&packed[o + 2..o + 4], &[0, 0], "block {b} scale padding");
        for (i, &q) in block.qs.iter().enumerate() {
            assert_eq!(packed[o + 4 + i] as i8, q, "block {b} value {i}");
        }
    }
    Ok(())
}
