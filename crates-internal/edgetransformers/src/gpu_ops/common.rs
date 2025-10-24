use crate::wgpu_context::WgpuContext;
use anyhow::Result;
use ndarray::{Array2, Array3};
use wgpu::Buffer;

pub async fn read_buffer_to_ndarray2d(
        context: &WgpuContext,
        buffer: &wgpu::Buffer,
        dims: (usize, usize),
    ) -> Result<Array2<f32>> {
        let (rows, cols) = dims;
        let buffer_size = (rows * cols * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    
    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
        context.queue.submit(std::iter::once(encoder.finish()));
    
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    
        context.device.poll(wgpu::PollType::wait_indefinitely());
    
        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
    
            let result_array = Array2::from_shape_vec((rows, cols), result_vec)?;
            Ok(result_array)
        } else {
            anyhow::bail!("Failed to read back 2D array from GPU")
        }
    }

pub async fn read_buffer_to_ndarray(
    context: &WgpuContext,
    buffer: &wgpu::Buffer,
    dims: (usize, usize, usize),
) -> Result<Array3<f32>> {
    let (batch, seq, hidden) = dims;
    let buffer_size = (batch * seq * hidden * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    context.device.poll(wgpu::PollType::wait_indefinitely());

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let result_array = Array3::from_shape_vec((batch, seq, hidden), result_vec)?;
        Ok(result_array)
    } else {
        anyhow::bail!("Failed to read back final result from GPU")
    }
}

/// A crucial helper for comparing floating-point vectors for near-equality.
/// Direct comparison `assert_eq!` will fail due to tiny precision differences
/// between CPU and GPU floating-point math.
pub fn assert_vecs_are_close(vec1: &[f32], vec2: &[f32], tolerance: f32) {
    assert_eq!(vec1.len(), vec2.len(), "Vectors have different lengths");
    for (i, (a, b)) in vec1.iter().zip(vec2.iter()).enumerate() {
        if (a - b).abs() > tolerance {
            panic!(
                "Mismatch at index {}: cpu = {}, gpu = {}. Difference: {}",
                i,
                a,
                b,
                (a - b).abs()
            );
        }
    }
}

