use crate::wgpu_context::WgpuContext;
use ndarray::{Array2, Array3};
use wgpu::util::DeviceExt;
use wgpu::PollType;


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulInfo {
    m: u32,
    k: u32,
    n: u32,
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

    assert!(m > 0 && k > 0 && n > 0, "Matrix dimensions for GPU matmul cannot be zero.");

    let a_cont = a.as_standard_layout();
    let b_cont = b.as_standard_layout();

    let shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matmul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
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
            timestamp_writes: None
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        // Calculate the number of workgroups to dispatch. shader uses 8x8 workgroups.
        let workgroup_x = (m as u32 + 7) / 8;
        let workgroup_y = (n as u32 + 7) / 8;
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

    // Request to map the staging buffer so we can read it.
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    
    context.device.poll(PollType::wait_indefinitely());

    // Await the mapping result and handle potential errors.
    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        // Don't forget to unmap the buffer
        drop(data);
        staging_buffer.unmap();

        // Convert the flat Vec<f32> back into our 2D ndarray::Array2
        Array2::from_shape_vec((m, n), result).unwrap()
    } else {
        panic!("Failed to read back data from GPU")
    }
}
