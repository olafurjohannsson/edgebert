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

/// Tests that the `run_gpu_reshape` and `run_gpu_unreshape` kernels are inverse operations.
/// It also verifies the reshape operation against a manual CPU implementation.
#[tokio::test]
async fn test_reshape_unreshape_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    let (b, s, h, d) = (2, 8, 4, 16); // batch, seq_len, num_heads, head_dim
    let hidden_size = h * d;

    let input_cpu: Array3<f32> = Array::random((b, s, hidden_size), Uniform::new(-1.0, 1.0));

    let buffer_size = (b * s * hidden_size * std::mem::size_of::<f32>()) as u64;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Reshape Input"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage,
    });
    let reshaped_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Reshaped Output"),
        size: buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let unreshaped_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Unreshaped Output"),
        size: buffer_size,
        usage,
        mapped_at_creation: false,
    });

    // --- 2. Act ---

    // --- Part A: Verify Reshape (Q/V layout) ---
    // CPU ground truth for reshape [B, S, H*D] -> [B, H, S, D]
    let mut cpu_reshaped = Array3::<f32>::zeros((b * h, s, d));
    for batch_idx in 0..b {
        for seq_idx in 0..s {
            for head_idx in 0..h {
                for head_dim_idx in 0..d {
                    let val = input_cpu[[batch_idx, seq_idx, head_idx * d + head_dim_idx]];
                    cpu_reshaped[[batch_idx * h + head_idx, seq_idx, head_dim_idx]] = val;
                }
            }
        }
    }

    let mut encoder1 =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let pipeline = compile_reshape_pipeline(&context);
    run_gpu_reshape(
        &context,
        &mut encoder1,
        &pipeline,
        &input_gpu,
        &reshaped_gpu,
        b as u32,
        s as u32,
        h as u32,
        d as u32,
        false,
    );
    context.queue.submit(std::iter::once(encoder1.finish()));

    let gpu_reshaped_array = read_buffer_to_ndarray(&context, &reshaped_gpu, (b * h, s, d)).await?;

    println!("Verifying Reshape GPU kernel against CPU implementation...");
    assert_vecs_are_close(
        cpu_reshaped.as_slice().unwrap(),
        gpu_reshaped_array.as_slice().unwrap(),
        1e-6,
    );
    println!("✅ Reshape GPU implementation is correct!");

    // --- Part B: Verify Unreshape is the inverse of Reshape ---
    let mut encoder2 =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let pipeline = compile_unreshape_pipeline(&context);
    run_gpu_unreshape(
        &context,
        &mut encoder2,
        &pipeline,
        &reshaped_gpu,
        &unreshaped_gpu,
        b as u32,
        s as u32,
        h as u32,
        d as u32,
    );
    context.queue.submit(std::iter::once(encoder2.finish()));

    let gpu_unreshaped_array =
        read_buffer_to_ndarray(&context, &unreshaped_gpu, (b, s, hidden_size)).await?;

    println!("Verifying Unreshape is the inverse of Reshape...");
    assert_vecs_are_close(
        input_cpu.as_slice().unwrap(),
        gpu_unreshaped_array.as_slice().unwrap(),
        1e-6,
    );
    println!("✅ Unreshape GPU implementation is correct!");

    Ok(())
}

#[tokio::test]
async fn test_reshape_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    let (b, s, h, d) = (2, 8, 4, 16); // batch, seq_len, num_heads, head_dim
    let hidden_size = h * d;

    let input_cpu: Array3<f32> = Array::random((b, s, hidden_size), Uniform::new(-1.0, 1.0));
    let buffer_size = (b * s * hidden_size * std::mem::size_of::<f32>()) as u64;

    let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Reshape Input"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Reshape Output"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- 2. Act ---

    // CPU ground truth for reshape [B, S, H*D] -> [B, H, S, D]
    // We will represent the output as [B*H, S, D] for easy comparison with a flat buffer.
    let mut cpu_reshaped = Array3::<f32>::zeros((b * h, s, d));
    for batch_idx in 0..b {
        for seq_idx in 0..s {
            for head_idx in 0..h {
                for head_dim_idx in 0..d {
                    let val = input_cpu[[batch_idx, seq_idx, head_idx * d + head_dim_idx]];
                    // The GPU output buffer is flat, so we calculate the flat index
                    let flat_output_idx =
                        batch_idx * (h * s * d) + head_idx * (s * d) + seq_idx * d + head_dim_idx;
                    // For verification, we put it into a structured ndarray
                    cpu_reshaped[[batch_idx * h + head_idx, seq_idx, head_dim_idx]] = val;
                }
            }
        }
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Reshape Test Encoder"),
    });
    let pipeline = compile_reshape_pipeline(&context);
    run_gpu_reshape(
        &context,
        &mut encoder,
        &pipeline,
        &input_gpu,
        &output_gpu,
        b as u32,
        s as u32,
        h as u32,
        d as u32,
        false,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    // The reshaped GPU buffer has a logical shape of [B, H, S, D], which is flat in memory.
    // For comparison, we can read it back as a 3D array of shape [B*H, S, D].
    let gpu_reshaped_array = read_buffer_to_ndarray(&context, &output_gpu, (b * h, s, d)).await?;

    // --- 3. Assert ---
    println!("Verifying Reshape GPU kernel against CPU implementation...");
    assert_vecs_are_close(
        cpu_reshaped.as_slice().unwrap(),
        gpu_reshaped_array.as_slice().unwrap(),
        1e-6,
    );
    println!("✅ Reshape GPU implementation is correct!");

    Ok(())
}

/// Tests that the `run_gpu_unreshape` kernel is the mathematical inverse of `run_gpu_reshape`.
#[tokio::test]
async fn test_unreshape_is_inverse_of_reshape() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    let (b, s, h, d) = (2, 8, 4, 16);
    let hidden_size = h * d;

    let input_cpu: Array3<f32> = Array::random((b, s, hidden_size), Uniform::new(-1.0, 1.0));
    let buffer_size = (b * s * hidden_size * std::mem::size_of::<f32>()) as u64;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    let original_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Unreshape Original"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage,
    });
    let reshaped_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Unreshape Intermediate"),
        size: buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let final_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Unreshape Final"),
        size: buffer_size,
        usage,
        mapped_at_creation: false,
    });

    // --- 2. Act ---
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Inverse Test Encoder"),
    });
    let pipeline = compile_reshape_pipeline(&context);
    // Run reshape: original -> reshaped
    run_gpu_reshape(
        &context,
        &mut encoder,
        &pipeline,
        &original_gpu,
        &reshaped_gpu,
        b as u32,
        s as u32,
        h as u32,
        d as u32,
        false,
    );
    let pipeline1 = compile_unreshape_pipeline(&context);
    // Run unreshape: reshaped -> final
    run_gpu_unreshape(
        &context,
        &mut encoder,
        &pipeline1,
        &reshaped_gpu,
        &final_gpu,
        b as u32,
        s as u32,
        h as u32,
        d as u32,
    );

    context.queue.submit(std::iter::once(encoder.finish()));

    let gpu_final_array = read_buffer_to_ndarray(&context, &final_gpu, (b, s, hidden_size)).await?;

    // --- 3. Assert ---
    println!("Verifying Unreshape is the inverse of Reshape...");
    // The final result should be identical to the original input.
    assert_vecs_are_close(
        input_cpu.as_slice().unwrap(),
        gpu_final_array.as_slice().unwrap(),
        1e-6,
    );
    println!("✅ Unreshape GPU implementation is correct!");

    Ok(())
}
