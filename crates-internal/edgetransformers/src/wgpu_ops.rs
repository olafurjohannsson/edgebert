use crate::wgpu_context::WgpuContext;
use ndarray::{Array1, Array2, Array3, s};
use wgpu::PollType;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FfnUniforms {
    pub m: u32, // sequence_length
    pub k: u32, // hidden_size
    pub n: u32, // intermediate_size
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulInfo {
    m: u32,
    k: u32,
    n: u32,
}

/// Runs a fused Feed-Forward Network kernel on the GPU for a 2D input.
pub async fn wgpu_feed_forward_2d(
    context: &WgpuContext,
    input: &Array2<f32>, // Shape: [seq_len, hidden_size]
    fc1_weight: &Array2<f32>,
    fc1_bias: &Array1<f32>,
    fc2_weight: &Array2<f32>,
    fc2_bias: &Array1<f32>,
) -> Array2<f32> {
    let (seq_len, hidden_size) = input.dim();
    let intermediate_size = fc1_weight.shape()[1];

    let shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FFN Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_ops/ffn/ffn.wgsl").into()),
        });

    let bind_group_layout =
        context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FFN Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniforms (FfnUniforms)
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
                    // Binding 1: Weights (FfnWeights, as a single storage buffer)
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
                    // Binding 2: Input tensor
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
                    // Binding 3: Output tensor
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

    let pipeline_layout = context
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Feedforward Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let compute_pipeline =
        context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Feedforward Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

    let uniforms = FfnUniforms {
        m: seq_len as u32,
        k: hidden_size as u32,
        n: intermediate_size as u32,
    };
    let uniform_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFN Uniforms Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let input_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFN Input Buffer"),
            contents: bytemuck::cast_slice(input.as_standard_layout().as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let output_buffer_size =
        (seq_len * hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("FFN Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut weights_data: Vec<f32> = Vec::new();
    weights_data.extend_from_slice(fc1_weight.as_standard_layout().as_slice().unwrap());
    weights_data.extend_from_slice(fc1_bias.as_slice().unwrap());
    weights_data.extend_from_slice(fc2_weight.as_standard_layout().as_slice().unwrap());
    weights_data.extend_from_slice(fc2_bias.as_slice().unwrap());

    let weights_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFN Weights Buffer"),
            contents: bytemuck::cast_slice(&weights_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FFN Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FFN Command Encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FFN Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_x = (seq_len as u32 + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
    }

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("FFN Staging Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    context.device.poll(PollType::wait_indefinitely());

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Reshape the flat vector back into a 2D array
        let result_array = Array2::from_shape_vec((seq_len, hidden_size), result_vec);
        result_array.unwrap()
    } else {
        panic!("Failed to read back data from GPU")
    }
}

pub async fn wgpu_matmul_3d_2d(
    context: &WgpuContext,
    a: &Array3<f32>,
    b: &Array2<f32>,
) -> Array3<f32> {
    let (batch_size, m, _) = a.dim();
    let n = b.shape()[1];
    let mut output = Array3::zeros((batch_size, m, n));

    for (i, a_slice) in a.axis_iter(ndarray::Axis(0)).enumerate() {
        let result_slice = wgpu_matmul_2d(context, &a_slice.to_owned(), b).await;
        output
            .slice_mut(ndarray::s![i, .., ..])
            .assign(&result_slice);
    }

    output
}

async fn wgpu_matmul_2d(context: &WgpuContext, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let n = b.shape()[1];

    assert!(
        m > 0 && k > 0 && n > 0,
        "Matrix dimensions for GPU matmul cannot be zero."
    );

    let a_cont = a.as_standard_layout();
    let b_cont = b.as_standard_layout();

    let shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matmul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_ops/matmul/matmul_tiled.wgsl").into()),
        });

    let info = MatmulInfo {
        m: m as u32,
        k: k as u32,
        n: n as u32,
    };
    let info_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Info Buffer"),
            contents: bytemuck::cast_slice(&[info]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // copy  data from ndarray to GPU.
    let a_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A Buffer"),
            contents: bytemuck::cast_slice(a_cont.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let b_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B Buffer"),
            contents: bytemuck::cast_slice(b_cont.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let c_buffer_size = (m * n * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let c_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C Buffer (Output)"),
        size: c_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, // COPY_SRC so we can read it back
        mapped_at_creation: false,
    });

    let bind_group_layout =
        context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Matmul Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer for matrix dimensions
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
                    // Binding 1: Read-only storage buffer for matrix A
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
                    // Binding 2: Read-only storage buffer for matrix B
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
                    // Binding 3: Read-write storage buffer for the output matrix C
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

    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c_buffer.as_entire_binding(),
                },
            ],
        });

    let pipeline_layout = context
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matmul Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let compute_pipeline =
        context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Matmul Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Matmul Encoder"),
        });

    {
        // Scoped to drop the mutable borrow of encoder
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Matmul Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        // Calculate the number of workgroups to dispatch. shader uses 8x8 workgroups.
        // let workgroup_x = (m as u32 + 7) / 8;
        // let workgroup_y = (n as u32 + 7) / 8;
        let workgroup_size = 16;
        let workgroup_x = (n as u32 + workgroup_size - 1) / workgroup_size;
        let workgroup_y = (m as u32 + workgroup_size - 1) / workgroup_size;

        compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: c_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, c_buffer_size);

    // Submit the commands to the GPU to execute.
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    context.device.poll(PollType::wait_indefinitely());

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        // Don't forget to unmap the buffer
        drop(data);
        staging_buffer.unmap();

        Array2::from_shape_vec((m, n), result).unwrap()
    } else {
        panic!("Failed to read back data from GPU")
    }
}
