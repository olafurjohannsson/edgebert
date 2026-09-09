//! WASM matmul implementations.
//! Single-threaded, uses WASM SIMD128 kernels.
//! Same public API as matmul_native.rs.

use ndarray::{Array2, ArrayView2};

use crate::cpu::kernels;

// ─── F32 ─────────────────────────────────────────────────────────

/// Computes C = A @ B^T for F32 matrices. Single-threaded WASM SIMD.
pub fn matmul_2d_cpu_f32(a: &ArrayView2<f32>, b_weights: &ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    assert_eq!(
        k, k2,
        "Matmul dimension mismatch: A[k]={} != B[k]={}",
        k, k2
    );

    let mut c = Array2::<f32>::zeros((m, n));

    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input must be contiguous");
    let b_slice = b_s.as_slice().expect("Weights must be contiguous");
    let c_slice = c.as_slice_mut().expect("Output must be contiguous");

    unsafe {
        kernels::wasm32::wasm_matmul_2d(c_slice, a_slice, b_slice, m, n, k);
    }

    c
}

/// Batched F32 matmul. In WASM, same as non-batched (no threading benefit).
pub fn matmul_2d_cpu_f32_batched(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
) -> Array2<f32> {
    let mut result = matmul_2d_cpu_f32(a, b_weights);
    if let Some(b) = bias {
        for mut row in result.rows_mut() {
            let row_slice = row.as_slice_mut().unwrap();
            for (val, &bias_val) in row_slice.iter_mut().zip(b.iter()) {
                *val += bias_val;
            }
        }
    }
    result
}

/// No-alloc F32 matmul.
pub fn matmul_2d_f32_noalloc(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    debug_assert_eq!(k, k2);
    debug_assert_eq!(output.dim(), (m, n));

    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input must be contiguous");
    let b_slice = b_s.as_slice().expect("Weights must be contiguous");
    let c_slice = output.as_slice_mut().expect("Output must be contiguous");

    unsafe {
        kernels::wasm32::wasm_matmul_2d(c_slice, a_slice, b_slice, m, n, k);
    }

    if let Some(bias) = bias {
        for row in 0..m {
            let row_start = row * n;
            for j in 0..n {
                c_slice[row_start + j] += bias[j];
            }
        }
    }
}

/// The faer path does not exist on wasm32: faer is a native dependency, and this
/// target has one thread and no AVX2 for it to exploit. The dispatch in
/// `LinearLayer::matmul_noalloc` is shared across targets, so the name has to
/// exist here; it delegates to the vector kernel, which is what wasm runs anyway.
pub fn matmul_2d_f32_faer_noalloc(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    matmul_2d_f32_noalloc(a, b_weights, bias, output);
}

/// No-alloc batched F32 matmul. In WASM, delegates to noalloc.
pub fn matmul_2d_f32_batched_noalloc(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    matmul_2d_f32_noalloc(a, b_weights, bias, output);
}

/// Faer-based matmul. Not available in WASM, falls back to SIMD.
pub fn matmul_2d_cpu_f32_faer(a: &ArrayView2<f32>, b_weights: &ArrayView2<f32>) -> Array2<f32> {
    matmul_2d_cpu_f32(a, b_weights)
}

// ─── BF16 ────────────────────────────────────────────────────────

use half::bf16;

/// Computes C = A @ B^T for F32 input and BF16 weights.
/// Converts BF16 to F32 per-row, then uses F32 SIMD kernel.
pub fn matmul_2d_cpu_bf16(a: &ArrayView2<f32>, b_weights: &ArrayView2<bf16>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    assert_eq!(k, k2, "Matmul dimension mismatch");

    let mut c = Array2::<f32>::zeros((m, n));

    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input must be contiguous");
    let b_slice = b_s.as_slice().expect("Weights must be contiguous");

    // Convert BF16 weights to F32 (one-time cost)
    let b_f32: Vec<f32> = b_slice.iter().map(|v| v.to_f32()).collect();

    let c_slice = c.as_slice_mut().expect("Output must be contiguous");

    unsafe {
        kernels::wasm32::wasm_matmul_2d(c_slice, a_slice, &b_f32, m, n, k);
    }

    c
}

// ─── Quantized (scalar fallback for now) ─────────────────────────

use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0};

/// Q8_0 matmul — scalar fallback for WASM.
pub fn matmul_2d_cpu_q8_0(a: &ArrayView2<f32>, b_weights: &[BlockQ8_0]) -> Array2<f32> {
    let (m, k) = a.dim();
    let k_per_block = 32;
    let n = (b_weights.len() * k_per_block) / k;

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();

    for row in 0..m {
        let a_row = &a_s.as_slice().unwrap()[row * k..(row + 1) * k];
        let out_row = &mut c.as_slice_mut().unwrap()[row * n..(row + 1) * n];

        // SIMD128 rather than the scalar fallback: this is the kernel the
        // browser chat runs for every token of every quantised model.
        unsafe {
            kernels::wasm32::wasm_matmul_vec_q8_0(out_row, a_row.as_ptr(), b_weights, k);
        }
    }

    c
}

/// Q4_K matmul — scalar fallback for WASM.
pub fn matmul_2d_cpu_q4_k(a: &ArrayView2<f32>, b_weights: &[BlockQ4_K]) -> Array2<f32> {
    let (m, k) = a.dim();
    let k_per_block = 256;
    let n = (b_weights.len() * k_per_block) / k;

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();

    // TODO: Add WASM SIMD Q4_K kernel
    // For now, use scalar
    for row in 0..m {
        let _a_row = &a_s.as_slice().unwrap()[row * k..(row + 1) * k];
        let out_row = &mut c.as_slice_mut().unwrap()[row * n..(row + 1) * n];

        for (j, out) in out_row.iter_mut().enumerate() {
            let num_blocks_per_row = k / k_per_block;
            let start = j * num_blocks_per_row;
            let end = start + num_blocks_per_row;
            let _ = &b_weights[start..end];
            // TODO: scalar q4_k dot product
            *out = 0.0;
        }
    }

    c
}

/// Q6_K scalar fallback for WASM.
pub fn matmul_2d_cpu_q6_k(a: &ArrayView2<f32>, b_weights: &[BlockQ6_K]) -> Array2<f32> {
    use crate::cpu::kernels::quantize::quantize_row_q8_k;
    use crate::cpu::kernels::scalar::vec_dot_q6k_q8k_scalar;

    let (m, k) = a.dim();
    let num_blocks_per_row = k / 256;
    let n = b_weights.len() / num_blocks_per_row;

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();

    for row in 0..m {
        let a_row = &a_s.as_slice().unwrap()[row * k..(row + 1) * k];
        let a_q8 = quantize_row_q8_k(a_row);
        let out_row = &mut c.as_slice_mut().unwrap()[row * n..(row + 1) * n];

        for (j, out) in out_row.iter_mut().enumerate() {
            let start = j * num_blocks_per_row;
            let end = start + num_blocks_per_row;
            *out = vec_dot_q6k_q8k_scalar(k, &b_weights[start..end], &a_q8);
        }
    }

    c
}

/// Q6_K matmul variant 2, same as q6_k for WASM.
pub fn matmul_2d_cpu_q6_k2(input: &ArrayView2<f32>, weights: &[BlockQ6_K]) -> Array2<f32> {
    matmul_2d_cpu_q6_k(input, weights)
}

/// wasm has no AVX2 kernel, so there is no column-parallel variant; the
/// row-parallel path is the only implementation.
pub fn matmul_2d_f32_noalloc_par_n(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    matmul_2d_f32_noalloc(a, b_weights, bias, output)
}

/// wasm has no AVX2 tile kernel; the vector path is the only implementation.
pub fn matmul_2d_f32_tile43_par_n(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    matmul_2d_f32_noalloc(a, b_weights, bias, output)
}
