use super::*;
use crate::gpu_ops::common::{assert_vecs_are_close, read_buffer_to_ndarray};
use crate::wgpu_context::WgpuContext;
use ndarray::{Array, Array1, Array2, Array3, s};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use std::sync::Arc;

use anyhow::Result;
use wgpu::util::DeviceExt;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}

#[tokio::test]
async fn test_apply_mask_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    let (batch_size, num_heads, seq_len) = (2, 4, 8);

    // Create a simple mask: first batch item allows all, second allows half.
    let mut mask_cpu = Array2::<f32>::ones((batch_size, seq_len));
    mask_cpu.slice_mut(s![1, seq_len / 2..]).fill(0.0);

    // Create initial scores (e.g., all zeros)
    let scores_cpu: Array3<f32> = Array3::zeros((batch_size * num_heads, seq_len, seq_len));

    let scores_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Mask Scores"),
        contents: bytemuck::cast_slice(scores_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let mask_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Mask"),
        contents: bytemuck::cast_slice(mask_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // --- 2. Act ---

    // CPU Ground Truth
    let mut cpu_result = scores_cpu.clone();
    let neg_inf = -1.0e9;
    for b in 0..batch_size {
        for h in 0..num_heads {
            for s_q in 0..seq_len {
                for s_k in 0..seq_len {
                    if mask_cpu[[b, s_k]] == 0.0 {
                        cpu_result[[b * num_heads + h, s_q, s_k]] = neg_inf;
                    }
                }
            }
        }
    }

    // GPU Path
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let pipeline = compile_apply_mask_pipeline(&context);
    run_gpu_apply_mask(
        &context,
        &mut encoder,
        &pipeline,
        &scores_gpu,
        &mask_gpu,
        batch_size as u32,
        num_heads as u32,
        seq_len as u32,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    let gpu_result_array = read_buffer_to_ndarray(
        &context,
        &scores_gpu,
        (batch_size * num_heads, seq_len, seq_len),
    )
    .await?;

    // --- 3. Assert ---
    println!("Verifying Apply Mask GPU kernel against CPU implementation...");
    assert_vecs_are_close(
        cpu_result.as_slice().unwrap(),
        gpu_result_array.as_slice().unwrap(),
        1e-6,
    );
    println!("✅ Apply Mask GPU implementation is correct!");

    Ok(())
}
