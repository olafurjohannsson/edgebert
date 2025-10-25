use super::*;
use crate::gpu_ops::common::{
    assert_vecs_are_close, read_buffer_to_ndarray, read_buffer_to_ndarray2d,
};
use crate::wgpu_context::WgpuContext;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use std::sync::Arc;

use anyhow::Result;
use wgpu::util::DeviceExt;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}

/// Tests that the `run_gpu_softmax` kernel produces the same result as a CPU implementation.
#[tokio::test]
async fn test_softmax_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    let rows = 4;
    let cols = 128;
    let scale = 0.125; // 1.0 / sqrt(64)

    let input_cpu: Array2<f32> = Array::random((rows, cols), Uniform::new(-5.0, 5.0));

    // GPU buffer is created with a copy of the data, as the kernel works in-place
    let data_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Softmax Data"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    // --- 2. Act ---

    // Calculate ground truth on the CPU
    let mut cpu_result = input_cpu.clone();
    for mut row in cpu_result.axis_iter_mut(ndarray::Axis(0)) {
        // Scale
        row *= scale;
        // Stable softmax: subtract max, exp, then normalize
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|x| (x - max_val).exp());
        let sum = row.sum();
        row /= sum;
    }

    // Run the GPU kernel
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Softmax Test Encoder"),
    });
    let pipeline = compile_softmax_pipeline(&context);
    run_gpu_softmax(
        &context,
        &mut encoder,
        &pipeline,
        &data_gpu,
        rows as u32,
        cols as u32,
        scale,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    let gpu_result_array = read_buffer_to_ndarray2d(&context, &data_gpu, (rows, cols)).await?;

    // --- 3. Assert ---
    println!("Verifying Softmax GPU kernel against CPU implementation...");
    assert_vecs_are_close(
        cpu_result.as_slice().unwrap(),
        gpu_result_array.as_slice().unwrap(),
        1e-5,
    );
    println!("✅ Softmax GPU implementation is correct!");

    Ok(())
}
