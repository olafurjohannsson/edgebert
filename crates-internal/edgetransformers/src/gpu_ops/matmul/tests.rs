use super::*;
use crate::gpu_ops::common::{
    assert_vecs_are_close, read_buffer_to_ndarray, read_buffer_to_ndarray2d,
};
use crate::wgpu_context::WgpuContext;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use std::sync::Arc;
use std::sync::Mutex;
use crate::bind_group::BindGroupCache;
use anyhow::Result;
use wgpu::util::DeviceExt;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}

#[tokio::test]
async fn test_matmul_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    // Test with non-square matrices to catch dimension errors
    let (m, k, n) = (32, 64, 48);

    let input_a_cpu: Array2<f32> = Array::random((m, k), Uniform::new(-1.0, 1.0));
    let input_b_cpu: Array2<f32> = Array::random((k, n), Uniform::new(-1.0, 1.0));

    let input_a_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Matmul A"),
        contents: bytemuck::cast_slice(input_a_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_b_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Matmul B"),
        contents: bytemuck::cast_slice(input_b_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_c_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Matmul C (Output)"),
        size: (m * n * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- 2. Act ---
    // Get the ground truth from the CPU `ndarray` implementation
    let cpu_result = input_a_cpu.dot(&input_b_cpu);

    // Run the GPU kernel
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Matmul Test Encoder"),
    });
    let pipeline = compile_matmul_pipeline(&context);
    run_gpu_matmul(
        &context,
        &mut encoder,
        &pipeline,
        &input_a_gpu,
        &input_b_gpu,
        &output_c_gpu,
        m as u32,
        k as u32,
        n as u32,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    // We need a 2D version of `read_buffer_to_ndarray`
    let gpu_result_array = read_buffer_to_ndarray2d(&context, &output_c_gpu, (m, n)).await?;

    // --- 3. Assert ---
    println!("Verifying Matmul GPU kernel against CPU implementation...");
    assert_vecs_are_close(
        cpu_result.as_slice().unwrap(),
        gpu_result_array.as_slice().unwrap(),
        1e-3, // Matmul can have slightly larger floating point differences
    );
    println!("✅ Matmul GPU implementation is correct!");

    Ok(())
}

