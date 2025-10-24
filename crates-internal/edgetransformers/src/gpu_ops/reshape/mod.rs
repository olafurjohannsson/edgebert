use wgpu::util::DeviceExt;
use wgpu::{Buffer, CommandEncoder, include_wgsl};
use crate::wgpu_context::WgpuContext;
use std::sync::Arc;


/// A self-contained helper to run the batch-aware `reshape.wgsl` shader.
///
/// This kernel is crucial for multi-head attention, permuting the dimensions of the
/// Q, K, and V projections to prepare them for batched matrix multiplication. It's the
/// GPU equivalent of `permute()` and is fully aware of the batch dimension.
pub fn run_gpu_reshape(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    input: &Buffer,
    output: &Buffer,
    b: u32, // batch_size
    s: u32, // seq_len
    h: u32, // num_heads
    d: u32, // head_dim
    transpose_k: bool,
) {
    /// A uniform struct to pass tensor dimensions to the reshape.wgsl shader.
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReshapeUniforms {
        b: u32,
        s: u32,
        h: u32,
        d: u32,
        transpose_k: u32,
        _padding: [u32; 3],
    }

    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./reshape.wgsl"));

    // Explicitly define the layout for the three bindings required by the shader.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Reshape Bind Group Layout"),
        entries: &[
            // @binding(0) var<uniform> uniforms
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
            // @binding(1) var<storage, read> input
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(2) var<storage, read_write> output
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
        label: Some("Reshape Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Reshape Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let uniforms = ReshapeUniforms {
        b,
        s,
        h,
        d,
        transpose_k: if transpose_k { 1 } else { 0 },
        _padding: [0; 3],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Reshape Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Reshape Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Reshape Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    // Dispatch a 3D grid of workgroups to cover the batch, sequence, and head dimensions.
    let workgroup_x = (s + 15) / 16; // Workgroup size is 16 in X dimension
    let workgroup_y = (h + 15) / 16; // Workgroup size is 16 in Y dimension
    let workgroup_z = b; // Dispatch one "layer" of workgroups for each item in the batch
    compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
}


/// A self-contained helper to run the `unreshape.wgsl` shader.
///
/// This kernel is the inverse of the reshape operation, taking the context vectors
/// from the multi-head format `[B, H, S, D]` and permuting them back into the
/// standard sequence format `[B, S, H*D]`, ready for the final output projection.
pub fn run_gpu_unreshape(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    input: &Buffer,
    output: &Buffer,
    b: u32, // batch_size
    s: u32, // seq_len
    h: u32, // num_heads
    d: u32, // head_dim
) {
    /// A uniform struct to pass tensor dimensions to the unreshape.wgsl shader.
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReshapeUniforms {
        b: u32,
        s: u32,
        h: u32,
        d: u32,
        _padding: [u32; 4],
    }

    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./unreshape.wgsl"));

    // Explicitly define the layout for the three bindings required by the shader.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Unreshape Bind Group Layout"),
        entries: &[
            // @binding(0) var<uniform> uniforms
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
            // @binding(1) var<storage, read> input
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(2) var<storage, read_write> output
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
        label: Some("Unreshape Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Unreshape Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let uniforms = ReshapeUniforms {
        b,
        s,
        h,
        d,
        _padding: [0; 4],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Unreshape Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Unreshape Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Unreshape Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    // Dispatch a 3D grid of workgroups to cover the batch, sequence, and head dimensions.
    let workgroup_x = (s + 15) / 16;
    let workgroup_y = (h + 15) / 16;
    let workgroup_z = b;
    compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
}

#[cfg(test)]
mod tests;