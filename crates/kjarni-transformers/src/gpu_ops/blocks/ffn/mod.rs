//! GPU-accelerated feed-forward network block.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2};
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, CommandEncoder, ComputePipeline};

use crate::WgpuContext;
use crate::activations::Activation;
use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::tensor::DType;
use crate::traits::FeedForwardLayout;
use crate::weights::ModelWeights;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FfnUniforms {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

struct FfnPipelines {
    fc1: (Arc<ComputePipeline>, Arc<BindGroupLayout>),
    fc2: (Arc<ComputePipeline>, Arc<BindGroupLayout>),
}

pub struct GpuFeedForwardWeights {
    pub(crate) fc1_weight: GpuTensor,
    pub(crate) fc1_bias: GpuTensor,
    pub(crate) fc2_weight: GpuTensor,
    pub(crate) fc2_bias: GpuTensor,
}

impl GpuFeedForwardWeights {
    pub fn new(
        fc1_weight: GpuTensor,
        fc1_bias: GpuTensor,
        fc2_weight: GpuTensor,
        fc2_bias: GpuTensor,
    ) -> Result<Self> {
        assert_eq!(fc1_weight.rank(), 2, "fc1 weight must be 2d");
        assert_eq!(fc2_weight.rank(), 2, "fc2 weight must be 2d");
        assert_eq!(fc1_bias.rank(), 1, "fc1 bias must be 1d");
        assert_eq!(fc2_bias.rank(), 1, "fc2 bias must be 1d");

        assert_eq!(fc1_weight.shape()[0], fc1_bias.shape()[0]);
        assert_eq!(fc2_weight.shape()[0], fc2_bias.shape()[0]);
        assert_eq!(fc1_weight.shape()[0], fc2_weight.shape()[1]);

        Ok(Self {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        })
    }

    pub fn from_layout(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        ffn_layout: &FeedForwardLayout,
        layer_idx: usize,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let i_str = &layer_idx.to_string();
        let label_prefix = format!("layer{}.ffn", layer_idx);

        let load_bias = |template: &Option<String>, label: &str| -> Result<GpuTensor> {
            let name = template.as_ref().ok_or_else(|| {
                anyhow!("standard ffn layout is missing required bias: {}", label)
            })?;
            GpuTensor::from_model_weights(
                context,
                weights,
                &name.replace("{}", i_str),
                target_dtype,
                &format!("{}.{}", label_prefix, label),
            )
        };

        let up_w = GpuTensor::from_model_weights(
            context,
            weights,
            &ffn_layout.up_weight.replace("{}", i_str),
            target_dtype,
            &format!("{}.up_w", label_prefix),
        )?;
        let down_w = GpuTensor::from_model_weights(
            context,
            weights,
            &ffn_layout.down_weight.replace("{}", i_str),
            target_dtype,
            &format!("{}.down_w", label_prefix),
        )?;
        let up_b = load_bias(&ffn_layout.up_bias, "up_b")?;
        let down_b = load_bias(&ffn_layout.down_bias, "down_b")?;

        Self::new(up_w, up_b, down_w, down_b)
    }

    pub fn from_ndarrays(
        context: &Arc<WgpuContext>,
        fc1_w_cpu: &Array2<f32>,
        fc1_b_cpu: &Array1<f32>,
        fc2_w_cpu: &Array2<f32>,
        fc2_b_cpu: &Array1<f32>,
    ) -> Result<Self> {
        assert_eq!(fc1_w_cpu.shape()[1], fc1_b_cpu.shape()[0]);
        assert_eq!(fc2_w_cpu.shape()[1], fc2_b_cpu.shape()[0]);
        assert_eq!(fc1_w_cpu.shape()[1], fc2_w_cpu.shape()[0]);

        let fc1_weight_transposed = GpuTensor::from_ndarray(context, &fc1_w_cpu.t().to_owned())?;
        let fc2_weight_transposed = GpuTensor::from_ndarray(context, &fc2_w_cpu.t().to_owned())?;
        let fc1_bias = GpuTensor::from_ndarray(context, fc1_b_cpu)?;
        let fc2_bias = GpuTensor::from_ndarray(context, fc2_b_cpu)?;

        Self::new(
            fc1_weight_transposed,
            fc1_bias,
            fc2_weight_transposed,
            fc2_bias,
        )
    }
}

pub struct GpuFeedForwardStd {
    pipelines: FfnPipelines,
    context: Arc<WgpuContext>,
}

impl GpuFeedForwardStd {
    pub fn new(context: &Arc<WgpuContext>, activation: Activation) -> Result<Self> {
        match activation {
            Activation::Gelu => (),
            Activation::GeluNew => (),
            Activation::Tanh => (),
            Activation::Relu => (),
            Activation::SilU => (),
            Activation::QuickGelu => (),
        }

        let (fc1_pipeline, fc1_layout) = compile_fc1_pipeline(context, activation);
        let (fc2_pipeline, fc2_layout) = compile_fc2_pipeline(context);

        Ok(Self {
            pipelines: FfnPipelines {
                fc1: (Arc::new(fc1_pipeline), Arc::new(fc1_layout)),
                fc2: (Arc::new(fc2_pipeline), Arc::new(fc2_layout)),
            },
            context: context.clone(),
        })
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weights: &GpuFeedForwardWeights,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let intermediate_size = weights.fc1_bias.shape()[0];
        let intermediate_shape = vec![input.shape()[0], input.shape()[1], intermediate_size];
        let intermediate = pool.get(intermediate_shape);

        let output = pool.get(input.shape().to_vec());

        self.run_fc1(encoder, weights, input, &intermediate);
        self.run_fc2(encoder, weights, &intermediate, &output);

        output
    }

    pub fn encode_2(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weights: &GpuFeedForwardWeights,
        pool: &mut GpuTensorPool,
        output: &GpuTensor,
    ) {
        let intermediate_size = weights.fc1_bias.shape()[0];
        let intermediate_shape = vec![input.shape()[0], input.shape()[1], intermediate_size];
        let intermediate = pool.get(intermediate_shape);

        self.run_fc1(encoder, weights, input, &intermediate);
        self.run_fc2(encoder, weights, &intermediate, output);
    }

    pub fn forward(
        &self,
        encoder: &mut CommandEncoder,
        weights: &GpuFeedForwardWeights,
        input: &GpuTensor,
        intermediate: &GpuTensor,
        output: &GpuTensor,
    ) {
        self.run_fc1(encoder, weights, input, intermediate);
        self.run_fc2(encoder, weights, intermediate, output);
    }

    pub fn run_fc1(
        &self,
        encoder: &mut CommandEncoder,
        weights: &GpuFeedForwardWeights,
        input: &GpuTensor,
        output: &GpuTensor,
    ) {
        let input_shape = input.shape();
        let output_shape = output.shape();
        let rows = (input_shape[0] * input_shape[1]) as u32;
        let hidden_size = input_shape[2] as u32;
        let intermediate_size = output_shape[2] as u32;

        let uniforms = FfnUniforms {
            m: rows,
            k: hidden_size,
            n: intermediate_size,
            _padding: 0,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ffn fc1 uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fc1 bind group"),
                layout: &self.pipelines.fc1.1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weights.fc1_weight.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weights.fc1_bias.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        let workgroups = (rows * intermediate_size).div_ceil(512);
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fc1 pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipelines.fc1.0);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    pub fn run_fc2(
        &self,
        encoder: &mut CommandEncoder,
        weights: &GpuFeedForwardWeights,
        input: &GpuTensor,
        output: &GpuTensor,
    ) {
        let input_shape = input.shape();
        let rows = (input_shape[0] * input_shape[1]) as u32;
        let intermediate_size = input_shape[2] as u32;
        let output_shape = output.shape();
        let hidden_size = output_shape[2] as u32;

        let uniforms = FfnUniforms {
            m: rows,
            k: intermediate_size,
            n: hidden_size,
            _padding: 0,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ffn fc2 uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fc2 bind group"),
                layout: &self.pipelines.fc2.1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weights.fc2_weight.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weights.fc2_bias.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        let workgroups = (rows * hidden_size).div_ceil(512);
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fc2 pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipelines.fc2.0);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

fn compile_fc1_pipeline(
    context: &WgpuContext,
    activation: Activation,
) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./fc1.wgsl"));

    let act_function = match activation {
        Activation::Gelu => 0.0,
        Activation::GeluNew => 1.0,
        Activation::Relu => 2.0,
        Activation::SilU => 3.0,
        Activation::Tanh => 4.0,
        // Must stay in step with the COMPUTE_ACT_TYPE branches in fc1.wgsl.
        Activation::QuickGelu => 5.0,
    };
    let constants = [("0", act_function)];

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fc1 bind group layout"),
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fc1 pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fc1 pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        cache: None,
    });

    (pipeline, bind_group_layout)
}

fn compile_fc2_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./fc2.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fc2 bind group layout"),
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fc2 pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fc2 pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    (pipeline, bind_group_layout)
}

#[cfg(test)]
mod tests;
