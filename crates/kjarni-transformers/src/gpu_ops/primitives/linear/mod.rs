use crate::gpu::GpuTensor;
use crate::tensor::DType;
use crate::{WgpuContext, gpu_profile};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LinearInfo {
    m: u32,
    k: u32,
    n: u32,
}

pub struct GpuLinearLayer {
    context: Arc<WgpuContext>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,

    // Pipelines
    gemv_f32: wgpu::ComputePipeline,
    gemv_bf16: wgpu::ComputePipeline,
    gemv_bf16_wide: wgpu::ComputePipeline,
    gemv_q4k: wgpu::ComputePipeline,
    bmm_q4k: wgpu::ComputePipeline,
    gemv_q6k: wgpu::ComputePipeline,
    bmm_q6k: wgpu::ComputePipeline,
    gemv_q8: wgpu::ComputePipeline,
    bmm_q8: wgpu::ComputePipeline,
    bmm_f32: wgpu::ComputePipeline,
    bmm_bf16: wgpu::ComputePipeline,

    buffer: wgpu::Buffer,
}

impl GpuLinearLayer {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let device = &context.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("./linear.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Linear Layer Layout"),
            entries: &[
                // 0: Info
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
                // 1: A (Input)
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
                // 2: B_F32 (Weights)
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
                // 3: B_BF16 (Weights)
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
                // 10: B_Q8 (Weights, repacked to 36 bytes per block)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9: B_Q6K (Weights, packed and padded to 212 bytes per block)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: B_Q4K (Weights, packed)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: Output
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
            label: Some("Linear Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Linear Dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            context: context.clone(),
            bind_group_layout: Arc::new(bind_group_layout),
            gemv_f32: make_pipeline("gemv_f32"),
            gemv_bf16: make_pipeline("gemv_bf16"),
            gemv_bf16_wide: make_pipeline("gemv_bf16_wide"),
            gemv_q4k: make_pipeline("gemv_q4k"),
            bmm_q4k: make_pipeline("bmm_q4k"),
            gemv_q6k: make_pipeline("gemv_q6k"),
            bmm_q6k: make_pipeline("bmm_q6k"),
            gemv_q8: make_pipeline("gemv_q8"),
            bmm_q8: make_pipeline("bmm_q8"),
            bmm_f32: make_pipeline("bmm_f32"),
            bmm_bf16: make_pipeline("bmm_bf16"),
            buffer,
        }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weights: &GpuTensor,
        output: &GpuTensor,
    ) {
        let m = (input.num_elements() / input.shape().last().unwrap()) as u32;
        let k = weights.shape()[1] as u32;
        let n = weights.shape()[0] as u32;

        let is_bf16 = weights.dtype() == DType::BF16;
        let is_gemv = m == 1;

        // Q4_K stays packed. Only the GEMV shape has a kernel for it; a batched
        // prefill still needs the weights expanded, which is why the loader only
        // keeps them packed when it can guarantee decode.
        let is_q4k = weights.dtype() == DType::Q4_K;
        let is_q6k = weights.dtype() == DType::Q6_K;
        let is_q8 = weights.dtype() == DType::Q8_0;
        let use_q4k_kernel = is_gemv && is_q4k;
        let use_q6k_kernel = is_gemv && is_q6k;
        let use_q8_kernel = is_gemv && is_q8;
        let use_wide_kernel = is_gemv && is_bf16 && n >= 128;

        let pipeline = if use_q4k_kernel {
            &self.gemv_q4k
        } else if is_q4k {
            &self.bmm_q4k
        } else if use_q6k_kernel {
            &self.gemv_q6k
        } else if is_q6k {
            &self.bmm_q6k
        } else if use_q8_kernel {
            &self.gemv_q8
        } else if is_q8 {
            &self.bmm_q8
        } else if use_wide_kernel {
            &self.gemv_bf16_wide
        } else {
            match (is_gemv, is_bf16) {
                (true, true) => &self.gemv_bf16,
                (true, false) => &self.gemv_f32,
                (false, true) => &self.bmm_bf16,
                (false, false) => &self.bmm_f32,
            }
        };

        let uniforms = LinearInfo { m, k, n };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Linear Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let (b_f32, b_bf16, b_q4k, b_q6k, b_q8) = if is_q4k {
            let b = &self.buffer;
            (b, b, weights.buffer(), b, b)
        } else if is_q6k {
            let b = &self.buffer;
            (b, b, b, weights.buffer(), b)
        } else if is_q8 {
            let b = &self.buffer;
            (b, b, b, b, weights.buffer())
        } else if is_bf16 {
            let b = &self.buffer;
            (b, weights.buffer(), b, b, b)
        } else {
            let b = &self.buffer;
            (weights.buffer(), b, b, b, b)
        };

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Linear BindGroup"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_f32.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_bf16.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: b_q4k.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: b_q6k.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: b_q8.as_entire_binding(),
                    },
                ],
            });

        let label = format!("Linear [{}x{} @ {}]", m, k, n);

        gpu_profile!(
            self.context,
            encoder,
            &label,
            |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);

                if use_q4k_kernel || use_q6k_kernel || use_q8_kernel {
                    // Each workgroup produces 4 output rows.
                    let rows = n.div_ceil(4);
                    let max_dim = 65535;
                    if rows > max_dim {
                        pass.dispatch_workgroups(max_dim, rows.div_ceil(max_dim), 1);
                    } else {
                        pass.dispatch_workgroups(rows, 1, 1);
                    }
                } else if use_wide_kernel {
                    // Dispatch 2D grid if N > 65535
                    let max_dim = 65535;
                    if n > max_dim {
                        let grid_x = max_dim;
                        let grid_y = n.div_ceil(max_dim);
                        pass.dispatch_workgroups(grid_x, grid_y, 1);
                    } else {
                        pass.dispatch_workgroups(n, 1, 1);
                    }
                } else if is_gemv {
                    // Standard GEMV: 1 Thread per Output Neuron
                    let groups = n.div_ceil(256);
                    pass.dispatch_workgroups(groups, 1, 1);
                } else if is_q4k || is_q6k || is_q8 {
                    // One workgroup reduces one output element.
                    pass.dispatch_workgroups(n, m, 1);
                } else {
                    // BMM: 2D Tiles
                    let groups_x = n.div_ceil(16);
                    let groups_y = m.div_ceil(16);
                    pass.dispatch_workgroups(groups_x, groups_y, 1);
                }
            }
        );
    }
}

#[cfg(test)]
mod tests;
