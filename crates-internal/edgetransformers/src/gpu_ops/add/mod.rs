use crate::wgpu_context::WgpuContext;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, CommandEncoder, ComputePipeline, include_wgsl};

pub fn compile_add_pipeline(context: &WgpuContext) -> ComputePipeline {
    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./add.wgsl"));
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Add Bind Group Layout"),
        entries: &[
            // @binding(0) var<uniform>
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
            // @binding(1) var<storage, read> a
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
            // @binding(2) var<storage, read> b
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
            // @binding(3) var<storage, read_write> output
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
        label: Some("Add Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout], // Use the layout we just created
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Add Pipeline"),
        layout: Some(&pipeline_layout), // Provide the explicit layout
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    pipeline
}

pub fn run_gpu_add(
    context: &WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    output: &wgpu::Buffer,
    size: u32,
) {
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct AddUniforms {
        size: u32,
        _padding: [u32; 3],
    } // Add padding for alignment
    let device = &context.device;
    let uniforms = AddUniforms {
        size,
        _padding: [0; 3],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Add Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Add Bind Group"),
        // The layout can be retrieved directly from the pre-compiled pipeline.
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
                resource: output.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Add Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    let workgroup_x = (size + 255) / 256;
    compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
}

#[cfg(test)]
mod tests;
