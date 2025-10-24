use super::*;
use crate::gpu_ops::common::{assert_vecs_are_close, read_buffer_to_ndarray};
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
async fn test_add_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    let batch_size = 2;
    let seq_len = 16;
    let hidden_size = 64;
    let total_elements = (batch_size * seq_len * hidden_size) as u32;

    let input_a_cpu: Array3<f32> =
        Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let input_b_cpu: Array3<f32> =
        Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));

    let input_a_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Add Input A"),
        contents: bytemuck::cast_slice(input_a_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_b_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Add Input B"),
        contents: bytemuck::cast_slice(input_b_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Add Output"),
        size: (total_elements as u64) * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- 2. Act ---
    let cpu_result = &input_a_cpu + &input_b_cpu;

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    run_gpu_add(
        &context,
        &mut encoder,
        &input_a_gpu,
        &input_b_gpu,
        &output_gpu,
        total_elements,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    let gpu_result_array =
        read_buffer_to_ndarray(&context, &output_gpu, (batch_size, seq_len, hidden_size)).await?;

    // --- 3. Assert ---
    println!("Verifying Add GPU kernel against CPU implementation...");
    assert_vecs_are_close(
        cpu_result.as_slice().unwrap(),
        gpu_result_array.as_slice().unwrap(),
        1e-6,
    );
    println!("✅ Add GPU implementation is correct!");

    Ok(())
}
