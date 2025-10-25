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

#[tokio::test]
async fn test_add_bias_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    let (rows, cols) = (4, 128);
    let total_elements = (rows * cols) as u32;

    let input_cpu: Array2<f32> = Array::random((rows, cols), Uniform::new(-1.0, 1.0));
    let bias_cpu: Array1<f32> = Array::random(cols, Uniform::new(-0.5, 0.5));

    let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test AddBias Input"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bias_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test AddBias Bias"),
        contents: bytemuck::cast_slice(bias_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test AddBias Output"),
        size: (total_elements as u64) * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- 2. Act ---
    // CPU ground truth: ndarray broadcasting handles this automatically.
    let cpu_result = &input_cpu + &bias_cpu;

    // Run the GPU kernel
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let pipeline = compile_add_bias_pipeline(&context);

    run_gpu_add_bias(
        &context,
        &mut encoder,
        &pipeline,
        &input_gpu,
        &bias_gpu,
        &output_gpu,
        total_elements,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    let gpu_result_array = read_buffer_to_ndarray2d(&context, &output_gpu, (rows, cols)).await?;

    // --- 3. Assert ---
    println!("Verifying AddBias GPU kernel against CPU implementation...");
    assert_vecs_are_close(
        cpu_result.as_slice().unwrap(),
        gpu_result_array.as_slice().unwrap(),
        1e-6,
    );
    println!("✅ AddBias GPU implementation is correct!");

    Ok(())
}
