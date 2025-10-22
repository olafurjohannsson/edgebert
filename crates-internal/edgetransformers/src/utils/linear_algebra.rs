//! Linear algebra operations for transformers

use ndarray::{Array1, Array2, Array3, Array4, Axis, Zip, s};

#[cfg(not(target_arch = "wasm32"))]
use ndarray::parallel::prelude::*;

use crate::wgpu_context::WgpuContext;
use crate::wgpu_ops;

pub async fn feed_forward_gpu(
    context: &WgpuContext,
    input: &Array3<f32>,
    fc1_weight: &Array2<f32>,
    fc1_bias: &Array1<f32>,
    fc2_weight: &Array2<f32>,
    fc2_bias: &Array1<f32>,
) -> Array3<f32> {
    let (batch_size, seq_len, hidden_size) = input.dim();
    let mut output = Array3::zeros((batch_size, seq_len, hidden_size));

    for i in 0..batch_size {
        let input_slice = input.slice(s![i, .., ..]);
        let result_slice = wgpu_ops::wgpu_feed_forward_2d(
            context,
            &input_slice.to_owned(),
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        )
        .await;
        output.slice_mut(s![i, .., ..]).assign(&result_slice);
    }

    output
}

pub async fn matmul_3d_2d_gpu(
    context: &WgpuContext,
    a: &Array3<f32>,
    b: &Array2<f32>,
) -> Array3<f32> {
    wgpu_ops::wgpu_matmul_3d_2d(context, a, b).await
}

/// Batched matrix multiplication: [batch, m, k] x [k, n] -> [batch, m, n]
#[inline(always)]
pub fn matmul_3d_2d(a: &Array3<f32>, b: &Array2<f32>) -> Array3<f32> {
    let (batch, m, k) = a.dim();
    let n = b.shape()[1];
    assert_eq!(
        k,
        b.shape()[0],
        "Matrix dimensions are incompatible for multiplication"
    );

    let a_cont = a.as_standard_layout();
    let b_cont = b.as_standard_layout();

    let mut c = Array3::<f32>::zeros((batch, m, n));

    #[cfg(not(target_arch = "wasm32"))]
    {
        if batch <= 4 {
            // Small batches: sequential
            Zip::from(c.axis_iter_mut(Axis(0)))
                .and(a_cont.axis_iter(Axis(0)))
                .for_each(|mut c_slice, a_slice| {
                    c_slice.assign(&a_slice.dot(&b_cont));
                });
        } else {
            // Large batches: parallel
            Zip::from(c.axis_iter_mut(Axis(0)))
                .and(a_cont.axis_iter(Axis(0)))
                .par_for_each(|mut c_slice, a_slice| {
                    c_slice.assign(&a_slice.dot(&b_cont));
                });
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        Zip::from(c.axis_iter_mut(Axis(0)))
            .and(a_cont.axis_iter(Axis(0)))
            .for_each(|mut c_slice, a_slice| {
                c_slice.assign(&a_slice.dot(&b_cont));
            });
    }

    c
}

/// 4D matrix multiplication for attention
#[inline(always)]
pub fn matmul_4d(a: &Array4<f32>, b: &Array4<f32>) -> Array4<f32> {
    assert_eq!(a.shape()[0], b.shape()[0], "Batch size mismatch");
    assert_eq!(a.shape()[1], b.shape()[1], "Heads mismatch");
    assert_eq!(a.shape()[3], b.shape()[2], "Dimension mismatch");

    let (batch, heads, seq1, dim) = a.dim();
    let seq2 = b.shape()[3];

    let a_cont = a.as_standard_layout();
    let b_cont = b.as_standard_layout();

    let a_3d = a_cont
        .view()
        .into_shape_with_order((batch * heads, seq1, dim))
        .expect("Could not reshape A into 3D");
    let b_3d = b_cont
        .view()
        .into_shape_with_order((batch * heads, dim, seq2))
        .expect("Could not reshape B into 3D");

    let total_batches = batch * heads;
    let mut output_3d = Array3::<f32>::zeros((total_batches, seq1, seq2));

    #[cfg(not(target_arch = "wasm32"))]
    {
        if total_batches <= 8 {
            Zip::from(output_3d.axis_iter_mut(Axis(0)))
                .and(a_3d.axis_iter(Axis(0)))
                .and(b_3d.axis_iter(Axis(0)))
                .for_each(|mut c_slice, a_slice, b_slice| {
                    let result = a_slice.dot(&b_slice);
                    c_slice.assign(&result);
                });
        } else {
            Zip::from(output_3d.axis_iter_mut(Axis(0)))
                .and(a_3d.axis_iter(Axis(0)))
                .and(b_3d.axis_iter(Axis(0)))
                .par_for_each(|mut c_slice, a_slice, b_slice| {
                    let result = a_slice.dot(&b_slice);
                    c_slice.assign(&result);
                });
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        Zip::from(output_3d.axis_iter_mut(Axis(0)))
            .and(a_3d.axis_iter(Axis(0)))
            .and(b_3d.axis_iter(Axis(0)))
            .for_each(|mut c_slice, a_slice, b_slice| {
                let result = a_slice.dot(&b_slice);
                c_slice.assign(&result);
            });
    }

    output_3d
        .into_shape_with_order((batch, heads, seq1, seq2))
        .unwrap()
}

/// Apply attention mask to scores
#[inline(always)]
pub fn apply_attention_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
    let mask_expanded = mask.clone().insert_axis(Axis(1)).insert_axis(Axis(2));

    if let Some(broadcast_mask) = mask_expanded.broadcast(scores.dim()) {
        Zip::from(&mut scores)
            .and(&broadcast_mask)
            .for_each(|s, &m| {
                if m == 0.0 {
                    *s = f32::NEG_INFINITY;
                }
            });
    }

    scores
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    const CHUNK: usize = 8;
    let n = a.len().min(b.len());

    // Split arrays for tree reduction
    let (chunks_a, tail_a) = a[..n].split_at(n - (n % CHUNK));
    let (chunks_b, tail_b) = b[..n].split_at(n - (n % CHUNK));

    // Tree-reduce dot product
    let chunk_dot: f32 = chunks_a
        .chunks_exact(CHUNK)
        .zip(chunks_b.chunks_exact(CHUNK))
        .map(|(ca, cb)| {
            let d0 = ca[0] * cb[0] + ca[1] * cb[1];
            let d1 = ca[2] * cb[2] + ca[3] * cb[3];
            let d2 = ca[4] * cb[4] + ca[5] * cb[5];
            let d3 = ca[6] * cb[6] + ca[7] * cb[7];
            (d0 + d1) + (d2 + d3)
        })
        .sum();

    let dot = chunk_dot + tail_a.iter().zip(tail_b).map(|(x, y)| x * y).sum::<f32>();

    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot / (norm_a * norm_b + 1e-8)
}
