use anyhow::Result;
use ndarray::{Array2, Array3};
use wgpu::util::DeviceExt;

use crate::WgpuContext;

pub async fn read_buffer_2d(
    context: &WgpuContext,
    buffer: &wgpu::Buffer,
    dims: (usize, usize),
) -> Result<Array2<f32>> {
    let (rows, cols) = dims;
    let buffer_size = (rows * cols * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging buffer 2d"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder 2d"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    crate::gpu_ops::context::drain(&context.device);

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let result_array = Array2::from_shape_vec((rows, cols), result_vec)?;
        Ok(result_array)
    } else {
        anyhow::bail!("failed to read back 2D array from GPU")
    }
}

/// Reads a GPU buffer as a 3D array. Blocks until GPU completes.
pub async fn read_buffer_3d(
    context: &WgpuContext,
    buffer: &wgpu::Buffer,
    dims: (usize, usize, usize),
) -> Result<Array3<f32>> {
    let (batch, seq, hidden) = dims;
    let buffer_size = (batch * seq * hidden * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging buffer 3d"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder 3d"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    crate::gpu_ops::context::drain(&context.device);

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let result_array = Array3::from_shape_vec((batch, seq, hidden), result_vec)?;
        Ok(result_array)
    } else {
        anyhow::bail!("failed to read back 3D array from GPU")
    }
}

/// Asserts two float slices are approximately equal within tolerance.
pub fn assert_vecs_are_close(vec1: &[f32], vec2: &[f32], tolerance: f32) {
    assert_eq!(
        vec1.len(),
        vec2.len(),
        "vectors have different lengths: {} vs {}",
        vec1.len(),
        vec2.len()
    );

    for (i, (&a, &b)) in vec1.iter().zip(vec2.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > tolerance {
            let start = i.saturating_sub(2);
            let end = (i + 3).min(vec1.len());

            panic!(
                "vectors differ at index {}:\n\
                 CPU value: {:.8}\n\
                 GPU value: {:.8}\n\
                 difference: {:.8} (tolerance: {:.8})\n\
                 context [{}-{}]:\n\
                   CPU: {:?}\n\
                   GPU: {:?}",
                i,
                a,
                b,
                diff,
                tolerance,
                start,
                end,
                &vec1[start..end],
                &vec2[start..end]
            );
        }
    }
}

/// Creates a GPU uniform buffer from a Pod type.
pub fn create_uniform_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    data: &T,
    label: &str,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&[*data]),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn get_test_context() -> Arc<WgpuContext> {
        WgpuContext::new().await.unwrap()
    }
    #[test]
    fn test_assert_vecs_close_pass() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.00001, 2.00001, 3.00001];
        assert_vecs_are_close(&a, &b, 1e-4);
    }

    #[test]
    #[should_panic(expected = "vectors differ at index")]
    fn test_assert_vecs_close_fail() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.1, 3.0];
        assert_vecs_are_close(&a, &b, 1e-5);
    }

    #[test]
    #[should_panic(expected = "different lengths")]
    fn test_assert_vecs_close_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_vecs_are_close(&a, &b, 1e-5);
    }

    #[test]
    fn test_assert_vecs_close_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_vecs_are_close(&a, &b, 1e-5);
    }

    #[test]
    fn test_assert_vecs_close_single_element() {
        let a = vec![42.0];
        let b = vec![42.0];
        assert_vecs_are_close(&a, &b, 1e-5);
    }

    #[test]
    fn test_assert_vecs_close_exact_match() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_vecs_are_close(&a, &b, 0.0); // Zero tolerance
    }

    #[test]
    fn test_assert_vecs_close_negative_values() {
        let a = vec![-1.0, -2.0, -3.0];
        let b = vec![-1.00001, -2.00001, -3.00001];
        assert_vecs_are_close(&a, &b, 1e-4);
    }

    #[test]
    fn test_assert_vecs_close_mixed_signs() {
        let a = vec![-1.0, 0.0, 1.0];
        let b = vec![-1.00001, 0.00001, 1.00001];
        assert_vecs_are_close(&a, &b, 1e-4);
    }

    #[test]
    fn test_assert_vecs_close_zeros() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        assert_vecs_are_close(&a, &b, 1e-10);
    }

    #[test]
    fn test_assert_vecs_close_large_values() {
        let a = vec![1e10, 2e10, 3e10];
        let b = vec![1e10 + 1.0, 2e10 + 1.0, 3e10 + 1.0];
        assert_vecs_are_close(&a, &b, 10.0);
    }

    #[test]
    fn test_assert_vecs_close_small_values() {
        let a = vec![1e-10, 2e-10, 3e-10];
        let b = vec![1e-10, 2e-10, 3e-10];
        assert_vecs_are_close(&a, &b, 1e-15);
    }

    #[test]
    fn test_assert_vecs_close_at_boundary() {
        let a = vec![1.0];
        let b = vec![1.0001];
        assert_vecs_are_close(&a, &b, 0.00011);
    }

    #[test]
    #[should_panic(expected = "vectors differ at index 0")]
    fn test_assert_vecs_close_just_over_boundary() {
        let a = vec![1.0];
        let b = vec![1.00011];
        assert_vecs_are_close(&a, &b, 0.0001);
    }

    #[test]
    #[should_panic(expected = "vectors differ at index 4")]
    fn test_assert_vecs_close_failure_at_end() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 999.0];
        assert_vecs_are_close(&a, &b, 1e-5);
    }

    #[test]
    #[should_panic(expected = "vectors differ at index 0")]
    fn test_assert_vecs_close_failure_at_start() {
        let a = vec![999.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_vecs_are_close(&a, &b, 1e-5);
    }

    #[test]
    fn test_assert_vecs_close_large_vector() {
        let a: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..10000).map(|i| i as f32 + 1e-8).collect();
        assert_vecs_are_close(&a, &b, 1e-5);
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestUniforms {
        value_a: f32,
        value_b: f32,
        value_c: u32,
        value_d: u32,
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_create_uniform_buffer_basic() {
        let context = get_test_context().await;

        let uniforms = TestUniforms {
            value_a: 1.0,
            value_b: 2.0,
            value_c: 3,
            value_d: 4,
        };

        let buffer = create_uniform_buffer(&context.device, &uniforms, "test uniforms");

        assert_eq!(buffer.size(), 16); // 4 x 4 bytes
        assert!(buffer.usage().contains(wgpu::BufferUsages::UNIFORM));
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_create_uniform_buffer_single_f32() {
        let context = get_test_context().await;

        let value: f32 = 42.0;
        let buffer = create_uniform_buffer(&context.device, &value, "single f32");

        assert_eq!(buffer.size(), 4);
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_create_uniform_buffer_single_u32() {
        let context = get_test_context().await;

        let value: u32 = 12345;
        let buffer = create_uniform_buffer(&context.device, &value, "single u32");

        assert_eq!(buffer.size(), 4);
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct LargeUniforms {
        values: [f32; 16],
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_create_uniform_buffer_larger_struct() {
        let context = get_test_context().await;

        let uniforms = LargeUniforms { values: [1.0; 16] };

        let buffer = create_uniform_buffer(&context.device, &uniforms, "large uniforms");

        assert_eq!(buffer.size(), 64); // 16 x 4 bytes
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_2d_simple() -> Result<()> {
        let context = get_test_context().await;

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_2d(&context, &buffer, (2, 3)).await?;

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[0, 2]], 3.0);
        assert_eq!(result[[1, 0]], 4.0);
        assert_eq!(result[[1, 1]], 5.0);
        assert_eq!(result[[1, 2]], 6.0);

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_2d_single_row() -> Result<()> {
        let context = get_test_context().await;

        let data = vec![10.0f32, 20.0, 30.0, 40.0];
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("single row"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_2d(&context, &buffer, (1, 4)).await?;

        assert_eq!(result.shape(), &[1, 4]);
        for i in 0..4 {
            assert_eq!(result[[0, i]], (i + 1) as f32 * 10.0);
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_2d_single_column() -> Result<()> {
        let context = get_test_context().await;

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("single column"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_2d(&context, &buffer, (4, 1)).await?;

        assert_eq!(result.shape(), &[4, 1]);
        for i in 0..4 {
            assert_eq!(result[[i, 0]], (i + 1) as f32);
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_2d_single_element() -> Result<()> {
        let context = get_test_context().await;

        let data = vec![42.0f32];
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("single element"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_2d(&context, &buffer, (1, 1)).await?;

        assert_eq!(result.shape(), &[1, 1]);
        assert_eq!(result[[0, 0]], 42.0);

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_2d_large() -> Result<()> {
        let context = get_test_context().await;

        let rows = 64;
        let cols = 128;
        let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();

        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("large buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_2d(&context, &buffer, (rows, cols)).await?;

        assert_eq!(result.shape(), &[rows, cols]);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(result[[i, j]], (i * cols + j) as f32);
            }
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_2d_negative_values() -> Result<()> {
        let context = get_test_context().await;

        let data = vec![-1.0f32, -2.0, -3.0, -4.0];
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("negative values"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_2d(&context, &buffer, (2, 2)).await?;

        assert_eq!(result[[0, 0]], -1.0);
        assert_eq!(result[[0, 1]], -2.0);
        assert_eq!(result[[1, 0]], -3.0);
        assert_eq!(result[[1, 1]], -4.0);

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_2d_zeros() -> Result<()> {
        let context = get_test_context().await;

        let data = vec![0.0f32; 9];
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("zeros"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_2d(&context, &buffer, (3, 3)).await?;

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(result[[i, j]], 0.0);
            }
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_3d_simple() -> Result<()> {
        let context = get_test_context().await;

        // Shape: [2, 3, 4] = 24 elements
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test buffer 3d"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_3d(&context, &buffer, (2, 3, 4)).await?;

        assert_eq!(result.shape(), &[2, 3, 4]);

        let mut idx = 0;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(result[[i, j, k]], idx as f32);
                    idx += 1;
                }
            }
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_3d_batch_1() -> Result<()> {
        let context = get_test_context().await;

        // Shape: [1, 4, 8] = 32 elements
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batch 1"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_3d(&context, &buffer, (1, 4, 8)).await?;

        assert_eq!(result.shape(), &[1, 4, 8]);

        for j in 0..4 {
            for k in 0..8 {
                let expected = (j * 8 + k) as f32 * 0.1;
                assert!((result[[0, j, k]] - expected).abs() < 1e-5);
            }
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_3d_single_element() -> Result<()> {
        let context = get_test_context().await;

        let data = vec![99.0f32];
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("single element 3d"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_3d(&context, &buffer, (1, 1, 1)).await?;

        assert_eq!(result.shape(), &[1, 1, 1]);
        assert_eq!(result[[0, 0, 0]], 99.0);

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_3d_transformer_shape() -> Result<()> {
        let context = get_test_context().await;

        let batch = 2;
        let seq = 16;
        let hidden = 64;
        let data: Vec<f32> = (0..(batch * seq * hidden))
            .map(|i| ((i % 100) as f32) / 10.0)
            .collect();

        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("transformer shape"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_3d(&context, &buffer, (batch, seq, hidden)).await?;

        assert_eq!(result.shape(), &[batch, seq, hidden]);

        assert!((result[[0, 0, 0]] - 0.0).abs() < 1e-5);
        assert!((result[[0, 0, 50]] - 5.0).abs() < 1e-5);
        assert!((result[[1, 0, 0]] - (((seq * hidden) % 100) as f32 / 10.0)).abs() < 1e-5);

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_3d_large() -> Result<()> {
        let context = get_test_context().await;

        let batch = 4;
        let seq = 32;
        let hidden = 128;
        let total = batch * seq * hidden;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();

        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("large 3d"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_3d(&context, &buffer, (batch, seq, hidden)).await?;

        assert_eq!(result.shape(), &[batch, seq, hidden]);

        assert_eq!(result[[0, 0, 0]], 0.0);
        assert_eq!(result[[batch - 1, seq - 1, hidden - 1]], (total - 1) as f32);

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_3d_negative_values() -> Result<()> {
        let context = get_test_context().await;

        let data: Vec<f32> = (0..8).map(|i| -(i as f32) - 1.0).collect();
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("negative 3d"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_3d(&context, &buffer, (2, 2, 2)).await?;

        assert_eq!(result[[0, 0, 0]], -1.0);
        assert_eq!(result[[0, 0, 1]], -2.0);
        assert_eq!(result[[1, 1, 1]], -8.0);

        Ok(())
    }
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_roundtrip_2d() -> Result<()> {
        let context = get_test_context().await;

        let original = Array2::from_shape_fn((8, 16), |(i, j)| (i * 16 + j) as f32);

        let data = original.as_slice().unwrap();
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("roundtrip 2d"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_2d(&context, &buffer, (8, 16)).await?;

        assert_eq!(original, result);

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_roundtrip_3d() -> Result<()> {
        let context = get_test_context().await;

        let original = Array3::from_shape_fn((4, 8, 16), |(i, j, k)| (i * 128 + j * 16 + k) as f32);

        let data = original.as_slice().unwrap();
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("roundtrip 3d"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_3d(&context, &buffer, (4, 8, 16)).await?;

        assert_eq!(original, result);

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_read_buffer_2d_special_floats() -> Result<()> {
        let context = get_test_context().await;

        let data = vec![0.0f32, -0.0, f32::MIN_POSITIVE, f32::MAX, f32::MIN];
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("special floats"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let result = read_buffer_2d(&context, &buffer, (1, 5)).await?;

        assert_eq!(result[[0, 0]], 0.0);
        assert_eq!(result[[0, 1]], -0.0);
        assert_eq!(result[[0, 2]], f32::MIN_POSITIVE);
        assert_eq!(result[[0, 3]], f32::MAX);
        assert_eq!(result[[0, 4]], f32::MIN);

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_multiple_reads_same_buffer() -> Result<()> {
        let context = get_test_context().await;

        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("multi read"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        // Read same buffer multiple times
        let result1 = read_buffer_2d(&context, &buffer, (3, 4)).await?;
        let result2 = read_buffer_2d(&context, &buffer, (3, 4)).await?;
        let result3 = read_buffer_2d(&context, &buffer, (3, 4)).await?;

        assert_eq!(result1, result2);
        assert_eq!(result2, result3);

        Ok(())
    }
}
