use crate::wgpu_context::WgpuContext;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, CommandEncoder, ComputePipeline, include_wgsl};

pub fn compile_softmax_pipeline(context: &WgpuContext) -> ComputePipeline {
    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./softmax.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Softmax Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Note: The buffer is read-write, so we declare it as such.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Softmax Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Softmax Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    pipeline
}

/// A self-contained helper to run the `softmax.wgsl` shader.
///
/// This kernel performs a numerically-stable softmax operation in-place on a buffer.
/// It also applies the attention scaling factor.
pub fn run_gpu_softmax(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    data: &Buffer, // The buffer to operate on in-place
    rows: u32,
    cols: u32,
    scale: f32,
) {
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct SoftmaxUniforms {
        rows: u32,
        cols: u32,
        scale: f32,
        _padding: u32,
    }

    let device = &context.device;

    let uniforms = SoftmaxUniforms {
        rows,
        cols,
        scale,
        _padding: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Softmax Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Softmax Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Softmax Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    // Dispatch one workgroup per row of the scores matrix
    let workgroup_x = (rows + 255) / 256;
    compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
}

#[cfg(test)]
mod tests;
