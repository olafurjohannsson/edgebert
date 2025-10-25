use super::*;
use crate::gpu_ops::common::{assert_vecs_are_close, read_buffer_to_ndarray};

use ndarray::{Array, Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use std::sync::Arc;
use crate::{LayerNorm, wgpu_context::WgpuContext, FeedForward};

use anyhow::Result;
use wgpu::util::DeviceExt;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}

/// Tests that the `run_gpu_layer_norm` kernel produces the same result as the CPU `LayerNorm`.
#[tokio::test]
async fn test_layer_norm_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;


    let batch_size = 1;
    let seq_len = 8;
    let hidden_size = 32;
    let eps = 1e-5;

    let input_cpu: Array3<f32> =
        Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let gamma_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(0.5, 1.5));
    let beta_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

    // Upload the same data to GPU buffers
    let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Input"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let gamma_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Gamma"),
        contents: bytemuck::cast_slice(gamma_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let beta_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Beta"),
        contents: bytemuck::cast_slice(beta_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Output"),
        size: (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- 2. Act: Run both CPU and GPU versions ---

    // Get the ground truth from the CPU implementation
    let cpu_layernorm = LayerNorm::new(gamma_cpu, beta_cpu, eps);
    let cpu_result = cpu_layernorm.forward_3d(&input_cpu);

    // Run the GPU kernel
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Test Encoder"),
    });
    let pipeline = compile_layer_norm_pipeline(&context);
    run_gpu_layer_norm(
        &context,
        &mut encoder,
        &pipeline,
        &input_gpu,
        &output_gpu,
        (batch_size * seq_len) as u32,
        hidden_size as u32,
        eps,
        &gamma_gpu,
        &beta_gpu,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    // Read the result back from the GPU
    let gpu_result_array =
        read_buffer_to_ndarray(&context, &output_gpu, (batch_size, seq_len, hidden_size)).await?;

    // --- 3. Assert: Compare the results ---

    println!("Verifying LayerNorm GPU kernel against CPU implementation...");
    assert_vecs_are_close(
        cpu_result.as_slice().unwrap(),
        gpu_result_array.as_slice().unwrap(),
        1e-4, // A reasonable tolerance for f32 differences between CPU/GPU
    );
    println!("✅ LayerNorm GPU implementation is correct!");

    Ok(())
}
