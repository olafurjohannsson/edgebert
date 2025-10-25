use crate::wgpu_context::WgpuContext;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, CommandEncoder, ComputePipeline, include_wgsl};

pub fn compile_apply_mask_pipeline(context: &WgpuContext) -> ComputePipeline {
    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./apply_mask.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Apply Mask Bind Group Layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Apply Mask Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Apply Mask Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    pipeline
}

/// A self-contained helper to run the `apply_mask.wgsl` shader.
///
/// This kernel applies the attention mask to the score matrix before the softmax
/// operation, setting scores for padding tokens to a large negative number.
pub fn run_gpu_apply_mask(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    scores: &Buffer, // The [B, H, S, S] score buffer, modified in-place
    mask: &Buffer,   // The [B, S] mask buffer
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
) {
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct MaskUniforms {
        batch_size: u32,
        num_heads: u32,
        seq_len: u32,
        _padding: u32,
    }

    let device = &context.device;

    let uniforms = MaskUniforms {
        batch_size,
        num_heads,
        seq_len,
        _padding: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Apply Mask Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Apply Mask Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scores.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: mask.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Apply Mask Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    // Dispatch a 3D grid to cover every element in the [B*H, S, S] score matrix
    let workgroup_x = (seq_len + 15) / 16;
    let workgroup_y = (seq_len + 15) / 16;
    let workgroup_z = batch_size * num_heads;
    compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
}

#[cfg(test)]
mod tests;
