use crate::wgpu_context::WgpuContext;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, CommandEncoder, ComputePipeline, include_wgsl};

pub fn compile_fc1_pipeline(context: &WgpuContext) -> ComputePipeline {
    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./fc1.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FC1 Bind Group Layout"),
        entries: &[
            // @binding(0) var<uniform> info: FfnUniforms;
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
            // @binding(1) var<storage, read> fc1_weight: array<f32>;
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
            // @binding(2) var<storage, read> fc1_bias: array<f32>;
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
            // @binding(3) var<storage, read> input: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(4) var<storage, read_write> output: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
        label: Some("FC1 Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FC1 Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub fn compile_fc2_pipeline(context: &WgpuContext) -> ComputePipeline {
    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./fc2.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FC2 Bind Group Layout"),
        entries: &[
            // @binding(0) var<uniform> info: FfnUniforms;
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
            // @binding(1) var<storage, read> fc2_weight: array<f32>;
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
            // @binding(2) var<storage, read> fc2_bias: array<f32>;
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
            // @binding(3) var<storage, read> input: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(4) var<storage, read_write> output: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
        label: Some("FC2 Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FC2 Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[derive(Clone)]
pub struct GpuFeedForwardWeights {
    pub fc1_weight: Arc<Buffer>,
    pub fc1_bias: Arc<Buffer>,
    pub fc2_weight: Arc<Buffer>,
    pub fc2_bias: Arc<Buffer>,
    pub norm_weight: Arc<Buffer>,
    pub norm_bias: Arc<Buffer>,
}

pub fn run_gpu_ffn(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    fc1_pipeline: &ComputePipeline,
    fc2_pipeline: &ComputePipeline,
    input: &Buffer,
    intermediate: &Buffer,  // Temp buffer for intermediate results
    output: &Buffer,
    weights: &GpuFeedForwardWeights,
    rows: u32,              // batch_size * seq_len
    hidden_size: u32,       // K
    intermediate_size: u32, // N
) {
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct FfnUniforms {
        m: u32,
        k: u32,
        n: u32,
        _padding: u32,
    }

    let device = &context.device;

    let uniforms = FfnUniforms {
        m: rows,
        k: hidden_size,
        n: intermediate_size,
        _padding: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("FFN Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // === FC1 Pass ===
    let fc1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FC1 Bind Group"),
        layout: &fc1_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weights.fc1_weight.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: weights.fc1_bias.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: intermediate.as_entire_binding(),
            },
        ],
    });

    let total_fc1_outputs = rows * intermediate_size;
    let workgroups_fc1 = (total_fc1_outputs + 511) / 512;

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FC1 Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(fc1_pipeline);
        compute_pass.set_bind_group(0, &fc1_bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroups_fc1, 1, 1);
    }

    // === FC2 Pass ===
    let fc2_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FC2 Bind Group"),
        layout: &fc2_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weights.fc2_weight.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: weights.fc2_bias.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: intermediate.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let total_fc2_outputs = rows * hidden_size;
    let workgroups_fc2 = (total_fc2_outputs + 511) / 512;

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FC2 Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(fc2_pipeline);
        compute_pass.set_bind_group(0, &fc2_bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroups_fc2, 1, 1);
    }
}

#[cfg(test)]
mod tests;

// use crate::wgpu_context::WgpuContext;
// use std::sync::Arc;
// use wgpu::util::DeviceExt;
// use wgpu::{Buffer, CommandEncoder, ComputePipeline, include_wgsl};

// pub fn compile_ffn_pipeline(context: &WgpuContext) -> ComputePipeline {
//     let device = &context.device;
//     let shader = device.create_shader_module(include_wgsl!("./ffn.wgsl"));

//     // Define the layout explicitly to match the shader's expectations.
//     let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//         label: Some("FFN Bind Group Layout"),
//         entries: &[
//             // @binding(0) var<uniform> info: FfnUniforms;
//             wgpu::BindGroupLayoutEntry {
//                 binding: 0,
//                 visibility: wgpu::ShaderStages::COMPUTE,
//                 ty: wgpu::BindingType::Buffer {
//                     ty: wgpu::BufferBindingType::Uniform,
//                     has_dynamic_offset: false,
//                     min_binding_size: None,
//                 },
//                 count: None,
//             },
//             // @binding(1) var<storage, read> weights: array<f32>;
//             wgpu::BindGroupLayoutEntry {
//                 binding: 1,
//                 visibility: wgpu::ShaderStages::COMPUTE,
//                 ty: wgpu::BindingType::Buffer {
//                     ty: wgpu::BufferBindingType::Storage { read_only: true },
//                     has_dynamic_offset: false,
//                     min_binding_size: None,
//                 },
//                 count: None,
//             },
//             // @binding(2) var<storage, read> input: array<f32>;
//             wgpu::BindGroupLayoutEntry {
//                 binding: 2,
//                 visibility: wgpu::ShaderStages::COMPUTE,
//                 ty: wgpu::BindingType::Buffer {
//                     ty: wgpu::BufferBindingType::Storage { read_only: true },
//                     has_dynamic_offset: false,
//                     min_binding_size: None,
//                 },
//                 count: None,
//             },
//             // @binding(3) var<storage, read_write> output: array<f32>;
//             wgpu::BindGroupLayoutEntry {
//                 binding: 3,
//                 visibility: wgpu::ShaderStages::COMPUTE,
//                 ty: wgpu::BindingType::Buffer {
//                     ty: wgpu::BufferBindingType::Storage { read_only: false },
//                     has_dynamic_offset: false,
//                     min_binding_size: None,
//                 },
//                 count: None,
//             },
//         ],
//     });

//     let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//         label: Some("FFN Pipeline Layout"),
//         bind_group_layouts: &[&bind_group_layout],
//         push_constant_ranges: &[],
//     });

//     let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//         label: Some("FFN Pipeline"),
//         layout: Some(&pipeline_layout),
//         module: &shader,
//         entry_point: Some("main"),
//         compilation_options: Default::default(),
//         cache: None,
//     });
//     pipeline
// }

// #[derive(Clone)]
// pub struct GpuFeedForwardWeights {
//     pub packed_weights: Arc<wgpu::Buffer>,
//     pub norm_weight: Arc<Buffer>,
//     pub norm_bias: Arc<Buffer>,
// }

// pub fn run_gpu_ffn(
//     context: &WgpuContext,
//     encoder: &mut CommandEncoder,
//     pipeline: &ComputePipeline,
//     input: &Buffer,
//     output: &Buffer,
//     weights: &GpuFeedForwardWeights,
//     rows: u32,              // batch_size * seq_len
//     hidden_size: u32,       // K
//     intermediate_size: u32, // N
// ) {
//     /// A uniform struct to pass metadata (dimensions) to the ffn.wgsl shader.
//     #[repr(C)]
//     #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
//     struct FfnUniforms {
//         m: u32,
//         k: u32,
//         n: u32,
//         _padding: u32, // Structs in WGSL are aligned to 16 bytes
//     }

//     let device = &context.device;

//     let uniforms = FfnUniforms {
//         m: rows,
//         k: hidden_size,
//         n: intermediate_size,
//         _padding: 0,
//     };
//     let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("FFN Uniforms"),
//         contents: bytemuck::cast_slice(&[uniforms]),
//         usage: wgpu::BufferUsages::UNIFORM,
//     });

//     let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
//         label: Some("FFN Bind Group"),
//         layout: &pipeline.get_bind_group_layout(0),
//         entries: &[
//             wgpu::BindGroupEntry {
//                 binding: 0,
//                 resource: uniform_buffer.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 1,
//                 resource: weights.packed_weights.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 2,
//                 resource: input.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 3,
//                 resource: output.as_entire_binding(),
//             },
//         ],
//     });

//     let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
//         label: Some("FFN Compute Pass"),
//         timestamp_writes: None,
//     });
//     compute_pass.set_pipeline(&pipeline);
//     compute_pass.set_bind_group(0, &bind_group, &[]);
//     // The shader processes one row (token) per thread.
    
//     // let workgroup_x = (rows + 255) / 256; // ISSUE 17 WORKSGROUPS = 4352 THREADS(256 threads each workgroup) spreading work accross 13 MILLION OPS
//     let total_outputs = rows * intermediate_size;
//     let workgroup_x = (total_outputs + 255) / 256;

//     compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
// }

// #[cfg(test)]
// mod tests;
