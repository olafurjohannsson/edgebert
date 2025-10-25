use super::*;
use crate::gpu_ops::common::{assert_vecs_are_close, read_buffer_to_ndarray};
use crate::{FeedForward, LayerNorm, wgpu_context::WgpuContext};
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use std::sync::Arc;

use anyhow::Result;
use wgpu::util::DeviceExt;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}

/// Tests that the `run_gpu_ffn` kernel produces the same result as the CPU `FeedForward`.
#[tokio::test]
async fn test_ffn_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    let batch_size = 1;
    let seq_len = 8;
    let hidden_size = 32;
    let intermediate_size = hidden_size * 4;

    let input_cpu: Array3<f32> =
        Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let intermediate_w_cpu: Array2<f32> =
        Array::random((hidden_size, intermediate_size), Uniform::new(-0.5, 0.5));
    let intermediate_b_cpu: Array1<f32> = Array::random(intermediate_size, Uniform::new(-0.5, 0.5));
    let output_w_cpu: Array2<f32> =
        Array::random((intermediate_size, hidden_size), Uniform::new(-0.5, 0.5));
    let output_b_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

    // Upload separate weights (no packing)
    let fc1_weight_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("FC1 Weight"),
        contents: bytemuck::cast_slice(intermediate_w_cpu.as_standard_layout().as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fc1_bias_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("FC1 Bias"),
        contents: bytemuck::cast_slice(intermediate_b_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fc2_weight_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("FC2 Weight"),
        contents: bytemuck::cast_slice(output_w_cpu.as_standard_layout().as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let fc2_bias_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("FC2 Bias"),
        contents: bytemuck::cast_slice(output_b_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test FFN Input"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Intermediate buffer between FC1 and FC2
    let intermediate_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test FFN Intermediate"),
        size: (batch_size * seq_len * intermediate_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test FFN Output"),
        size: (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dummy_norm_w_cpu: Array1<f32> = Array1::zeros(hidden_size);
    let dummy_norm_b_cpu: Array1<f32> = Array1::zeros(hidden_size);

    let dummy_norm_w_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dummy Norm W"),
        contents: bytemuck::cast_slice(dummy_norm_w_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dummy_norm_b_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dummy Norm B"),
        contents: bytemuck::cast_slice(dummy_norm_b_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let gpu_weights = GpuFeedForwardWeights {
        fc1_weight: Arc::new(fc1_weight_gpu),
        fc1_bias: Arc::new(fc1_bias_gpu),
        fc2_weight: Arc::new(fc2_weight_gpu),
        fc2_bias: Arc::new(fc2_bias_gpu),
        norm_weight: Arc::new(dummy_norm_w_gpu),
        norm_bias: Arc::new(dummy_norm_b_gpu),
    };

    // CPU computation
    let cpu_ffn = FeedForward::new(
        intermediate_w_cpu.as_standard_layout().to_owned(),
        intermediate_b_cpu.clone(),
        output_w_cpu.as_standard_layout().to_owned(),
        output_b_cpu.clone(),
    );
    let cpu_result = cpu_ffn.forward(&input_cpu)?;

    // GPU computation
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("FFN Test Encoder"),
    });

    let fc1_pipeline = compile_fc1_pipeline(&context);
    let fc2_pipeline = compile_fc2_pipeline(&context);

    run_gpu_ffn(
        &context,
        &mut encoder,
        &fc1_pipeline,
        &fc2_pipeline,
        &input_gpu,
        &intermediate_gpu,
        &output_gpu,
        &gpu_weights,
        (batch_size * seq_len) as u32,
        hidden_size as u32,
        intermediate_size as u32,
    );
    context.queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely());

    let gpu_result_array =
        read_buffer_to_ndarray(&context, &output_gpu, (batch_size, seq_len, hidden_size)).await?;

    // --- 3. Assert ---
    println!("Verifying FFN GPU kernel against CPU implementation...");
    assert_vecs_are_close(
        cpu_result.as_slice().unwrap(),
        gpu_result_array.as_slice().unwrap(),
        1e-4,
    );
    println!("✅ FFN GPU implementation is correct!");

    Ok(())
}