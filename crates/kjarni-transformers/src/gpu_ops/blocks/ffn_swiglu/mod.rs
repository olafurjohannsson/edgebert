use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::gpu_ops::primitives::linear::GpuLinearLayer;
use crate::tensor::DType;
use crate::{WgpuContext, gpu_profile};
use anyhow::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedInfo {
    m: u32,
    k: u32,
    n: u32,
}

/// Holds the weight tensors for the GpuSwiGLUFFN operation.
pub struct GpuSwiGLUFFNWeights {
    pub(crate) gate_proj: GpuTensor,
    pub(crate) up_proj: GpuTensor,
    pub(crate) down_proj: GpuTensor,
}

impl GpuSwiGLUFFNWeights {
    pub fn new(gate_proj: GpuTensor, up_proj: GpuTensor, down_proj: GpuTensor) -> Result<Self> {
        assert_eq!(gate_proj.rank(), 2, "gate_proj weight must be 2D");
        assert_eq!(up_proj.rank(), 2, "up_proj weight must be 2D");
        assert_eq!(down_proj.rank(), 2, "down_proj weight must be 2D");
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

/// A GPU kernel for the SwiGLU Feed-Forward Network used in LLaMA.
pub struct GpuSwiGLUFFN {
    fused_bind_group_layout_bf16: wgpu::BindGroupLayout,
    fused_bind_group_layout_f32: wgpu::BindGroupLayout,
    fused_bind_group_layout_q4k: wgpu::BindGroupLayout,
    fused_gemv_bf16: wgpu::ComputePipeline,
    fused_bmm_bf16: wgpu::ComputePipeline,
    fused_gemv_f32: wgpu::ComputePipeline,
    fused_bmm_f32: wgpu::ComputePipeline,
    fused_gemv_q4k: wgpu::ComputePipeline,
    fused_bmm_q4k: wgpu::ComputePipeline,
    /// Fallback for weight dtypes the fused kernels do not read. See `encode`.
    elementwise_bind_group_layout: wgpu::BindGroupLayout,
    elementwise_pipeline: wgpu::ComputePipeline,
    linear_layer: GpuLinearLayer,
    context: Arc<WgpuContext>,
}

impl GpuSwiGLUFFN {
    pub fn new(context: &Arc<WgpuContext>) -> Result<Self> {
        let device = &context.device;

        let elementwise_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SwiGLU Elementwise Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let elementwise_shader = device.create_shader_module(wgpu::include_wgsl!("./swiglu.wgsl"));
        let elementwise_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SwiGLU Elementwise"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SwiGLU Elementwise Layout"),
                        bind_group_layouts: &[&elementwise_bind_group_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &elementwise_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let shader = device.create_shader_module(wgpu::include_wgsl!("./swiglu_fused.wgsl"));

        // BF16 layout: uniform, input, gate_bf16, up_bf16, output
        let fused_bind_group_layout_bf16 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fused SwiGLU BF16 Layout"),
                entries: &[
                    // 0: Uniforms
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
                    // 1: Input
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
                    // 2: gate_w (BF16)
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
                    // 3: up_w (BF16)
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

        // F32 layout: uniform, input, gate_f32, up_f32, output (reusing bindings 5,6 not needed - just different layout)
        let fused_bind_group_layout_f32 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fused SwiGLU F32 Layout"),
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
                    // For F32, we use bindings 5,6 in shader but map them to 2,3 in layout
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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

        // Q4_K uses bindings 7 and 8 for the two packed weight buffers.
        let fused_bind_group_layout_q4k =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fused SwiGLU Q4_K Layout"),
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
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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

        let layout_q4k = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fused SwiGLU Q4_K Pipeline Layout"),
            bind_group_layouts: &[&fused_bind_group_layout_q4k],
            push_constant_ranges: &[],
        });

        let layout_bf16 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fused SwiGLU BF16 Pipeline Layout"),
            bind_group_layouts: &[&fused_bind_group_layout_bf16],
            push_constant_ranges: &[],
        });

        let layout_f32 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fused SwiGLU F32 Pipeline Layout"),
            bind_group_layouts: &[&fused_bind_group_layout_f32],
            push_constant_ranges: &[],
        });

        let make_pipeline = |layout: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let _buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SwiGLU Dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Ok(Self {
            fused_bind_group_layout_bf16,
            fused_bind_group_layout_f32,
            fused_bind_group_layout_q4k,
            fused_gemv_bf16: make_pipeline(&layout_bf16, "fused_gemv_bf16"),
            fused_bmm_bf16: make_pipeline(&layout_bf16, "fused_bmm_bf16"),
            fused_gemv_f32: make_pipeline(&layout_f32, "fused_gemv_f32"),
            fused_bmm_f32: make_pipeline(&layout_f32, "fused_bmm_f32"),
            fused_gemv_q4k: make_pipeline(&layout_q4k, "fused_gemv_q4k"),
            fused_bmm_q4k: make_pipeline(&layout_q4k, "fused_bmm_q4k"),
            elementwise_bind_group_layout,
            elementwise_pipeline,
            linear_layer: GpuLinearLayer::new(context),
            context: context.clone(),
        })
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &GpuSwiGLUFFNWeights,
        input: &GpuTensor,
        output: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) {
        let rank = input.rank();
        assert_eq!(rank, 2, "Input tensor for GpuSwiGLUFFN must be 2D");

        let rows = input.shape()[0];
        let intermediate_size = weights.up_proj.shape()[0];

        // Step 1: gate/up projection with SiLU. Output: [rows, intermediate_size]
        let intermediate = pool.get(vec![rows, intermediate_size]);

        // The fused kernels read BF16, F32 and Q4_K only. Anything else -- Q8_0, the
        // encoding a `.kjq` decoder uses -- would be reinterpreted as bf16 and yield
        // NaN with no error, so those take the unfused route instead: two ordinary
        // linear projections through `GpuLinearLayer`, which dispatches on dtype
        // properly, then the elementwise activation.
        let dt = weights.gate_proj.dtype();
        let fused_reads = matches!(dt, DType::BF16 | DType::F32 | DType::Q4_K);
        if fused_reads {
            self.encode_fused_gate_up(encoder, input, weights, &intermediate);
        } else {
            let gate = pool.get(vec![rows, intermediate_size]);
            let up = pool.get(vec![rows, intermediate_size]);
            self.linear_layer
                .encode(encoder, input, &weights.gate_proj, &gate);
            self.linear_layer
                .encode(encoder, input, &weights.up_proj, &up);
            self.encode_swiglu_elementwise(encoder, &gate, &up, &intermediate);
        }

        self.linear_layer
            .encode(encoder, &intermediate, &weights.down_proj, output);
    }
    /// `silu(gate) * up`, elementwise. The unfused half of `encode`.
    fn encode_swiglu_elementwise(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate: &GpuTensor,
        up: &GpuTensor,
        output: &GpuTensor,
    ) {
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SwiGLU Elementwise BindGroup"),
                layout: &self.elementwise_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gate.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: up.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        let elements: usize = output.shape().iter().product();
        gpu_profile!(
            self.context,
            encoder,
            "SwiGLU elementwise",
            |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_pipeline(&self.elementwise_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((elements as u32).div_ceil(256), 1, 1);
            }
        );
    }

    fn encode_fused_gate_up(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weights: &GpuSwiGLUFFNWeights,
        output: &GpuTensor,
    ) {
        // The dispatch below falls back to the BF16 bindings for anything it does not
        // recognise, which reinterprets the bytes rather than failing. `encode` filters
        // on the same list; this catches a dtype added there and forgotten here.
        debug_assert!(
            matches!(
                weights.gate_proj.dtype(),
                DType::BF16 | DType::F32 | DType::Q4_K
            ),
            "fused SwiGLU cannot read {:?}; route it through the unfused path",
            weights.gate_proj.dtype()
        );
        let m = input.shape()[0] as u32;
        let k = input.shape()[1] as u32;
        let n = output.shape()[1] as u32;

        let is_bf16 = weights.gate_proj.dtype() == DType::BF16;
        let is_gemv = m == 1;

        let uniforms = FusedInfo { m, k, n };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Fused SwiGLU Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let is_q4k = weights.gate_proj.dtype() == DType::Q4_K;

        let (pipeline, bind_group) = if is_q4k {
            let pipeline = if is_gemv {
                &self.fused_gemv_q4k
            } else {
                &self.fused_bmm_q4k
            };
            let bind_group = self
                .context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Fused SwiGLU Q4_K BindGroup"),
                    layout: &self.fused_bind_group_layout_q4k,
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
                            binding: 7,
                            resource: weights.gate_proj.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: weights.up_proj.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: output.buffer().as_entire_binding(),
                        },
                    ],
                });
            (pipeline, bind_group)
        } else if is_bf16 {
            let pipeline = if is_gemv {
                &self.fused_gemv_bf16
            } else {
                &self.fused_bmm_bf16
            };
            let bind_group = self
                .context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Fused SwiGLU BF16 BindGroup"),
                    layout: &self.fused_bind_group_layout_bf16,
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
                            resource: weights.gate_proj.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: weights.up_proj.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: output.buffer().as_entire_binding(),
                        },
                    ],
                });
            (pipeline, bind_group)
        } else {
            let pipeline = if is_gemv {
                &self.fused_gemv_f32
            } else {
                &self.fused_bmm_f32
            };
            let bind_group = self
                .context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Fused SwiGLU F32 BindGroup"),
                    layout: &self.fused_bind_group_layout_f32,
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
                            binding: 5,
                            resource: weights.gate_proj.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: weights.up_proj.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: output.buffer().as_entire_binding(),
                        },
                    ],
                });
            (pipeline, bind_group)
        };

        let label = format!("Fused SwiGLU [{}x{} -> {}]", m, k, n);
        gpu_profile!(
            self.context,
            encoder,
            &label,
            |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);

                if is_gemv && is_q4k {
                    // Each workgroup produces 2 output rows.
                    pass.dispatch_workgroups(n.div_ceil(4), 1, 1);
                } else if is_gemv {
                    pass.dispatch_workgroups(n, 1, 1);
                } else if is_q4k {
                    // 256-wide workgroup reducing one output element, not a 16x16 tile.
                    pass.dispatch_workgroups(n, m, 1);
                } else {
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
