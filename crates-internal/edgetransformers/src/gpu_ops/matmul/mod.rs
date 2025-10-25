use crate::wgpu_context::WgpuContext;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, Buffer, CommandEncoder, ComputePipeline, Device, Queue, include_wgsl};

use crate::bind_group::BindGroupCache;

pub fn compile_matmul_pipeline(context: &WgpuContext) -> ComputePipeline {
    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./matmul_tiled.wgsl"));

    // Explicitly define the layout to match the shader.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Matmul Bind Group Layout"),
        entries: &[
            // @binding(0) Uniforms
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
            // @binding(1) Input Matrix A
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
            // @binding(2) Input Matrix B
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
            // @binding(3) Output Matrix C
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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
        label: Some("Matmul Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Matmul Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    pipeline
}

pub fn compile_bmm_pipeline(context: &WgpuContext) -> ComputePipeline {
    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./bmm.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("BMM Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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
        label: Some("BMM Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("BMM Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    pipeline
}

/// A uniform struct to pass matrix dimensions to the bmm.wgsl shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulInfo {
    b: u32,
    m: u32,
    k: u32,
    n: u32,
}

/// A helper function to run a batched 2D matrix multiplication `C = A * B` on the GPU.
///
/// This kernel is essential for attention score calculation. It treats the inputs
/// as batches of 2D matrices and performs the multiplication for each item in the batch.
/// - A shape: [batch, m, k]
/// - B shape: [batch, k, n]
/// - C shape: [batch, m, n]
pub fn run_gpu_bmm(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
) {
    let device = &context.device;
    let uniforms = MatmulInfo { b: batch, m, k, n };
    
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BMM Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BMM Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: c.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("BMM Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    const TILE_DIM: u32 = 32;
    let workgroup_x = (n + TILE_DIM - 1) / TILE_DIM;
    let workgroup_y = (m + TILE_DIM - 1) / TILE_DIM;
    let workgroup_z = batch;

    compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
}

/// Uniform struct for standard matrix multiplication
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulUniforms {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

/// Run 2D matrix multiplication: C = A * B
/// - A shape: [m, k]
/// - B shape: [k, n]
/// - C shape: [m, n]
pub fn run_gpu_matmul(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    m: u32,
    k: u32,
    n: u32,
) {
    let device = &context.device;
    let uniforms = MatmulUniforms {
        m,
        k,
        n,
        _padding: 0,
    };
    
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matmul Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Matmul Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: c.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Matmul Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    const TILE_DIM: u32 = 32;
    let workgroup_x = (n + TILE_DIM - 1) / TILE_DIM;
    let workgroup_y = (m + TILE_DIM - 1) / TILE_DIM;

    compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
}

#[cfg(test)]
mod tests;