//! A reusable orchestrator for running a generic transformer encoder pipeline on the GPU.

use anyhow::{Result, anyhow};
use ndarray::Array3;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, CommandEncoder, include_wgsl};

use crate::traits::TransformerConfig;
use crate::wgpu_context::WgpuContext;

// --- GPU-Native Data Structures ---
// These are just containers for buffer handles.
#[derive(Clone)]
pub struct GpuAttentionWeights {
    pub q_weight: Arc<Buffer>,
    pub q_bias: Arc<Buffer>,
    pub k_weight: Arc<Buffer>,
    pub k_bias: Arc<Buffer>,
    pub v_weight: Arc<Buffer>,
    pub v_bias: Arc<Buffer>,
    pub output_weight: Arc<Buffer>,
    pub output_bias: Arc<Buffer>,
    pub norm_weight: Arc<Buffer>,
    pub norm_bias: Arc<Buffer>,
}

#[derive(Clone)]
pub struct GpuFeedForwardWeights {
    // pub intermediate_weight: Arc<Buffer>,
    // pub intermediate_bias: Arc<Buffer>,
    // pub output_weight: Arc<Buffer>,
    // pub output_bias: Arc<Buffer>,
    pub packed_weights: Arc<wgpu::Buffer>,
    pub norm_weight: Arc<Buffer>,
    pub norm_bias: Arc<Buffer>,
}

#[derive(Clone)]
pub struct GpuTransformerLayer {
    pub attention_weights: GpuAttentionWeights,
    pub ffn_weights: GpuFeedForwardWeights,
}

/// The reusable orchestrator for the GPU encoder pipeline.
pub struct GpuEncoderPipeline {
    context: Arc<WgpuContext>,
    // You would cache the compiled pipelines here for performance
}

impl GpuEncoderPipeline {
    pub fn new(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self { context })
    }

    /// Executes the full, end-to-end GPU forward pass for a transformer encoder.
    pub async fn forward<C: TransformerConfig + ?Sized>(
        &self,
        config: &C,
        initial_embeddings: &Array3<f32>,
        embedding_norm_weights: (&Buffer, &Buffer),
        layers: &[GpuTransformerLayer],
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len, hidden_size) = initial_embeddings.dim();
        let device = &self.context.device;
        let buffer_size = (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>())
            as wgpu::BufferAddress;

        // Create the three main buffers for our pipeline
        let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pipeline Buffer A"),
            contents: bytemuck::cast_slice(
                initial_embeddings.as_standard_layout().as_slice().unwrap(),
            ),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline Buffer B"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let residual_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Residual Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder Pipeline"),
        });

        // Initial LayerNorm: Input: A, Output: B.
        run_gpu_layer_norm(
            &self.context,
            &mut encoder,
            &buffer_a,
            &buffer_b,
            (batch_size * seq_len) as u32,
            hidden_size as u32,
            config.layer_norm_eps(),
            embedding_norm_weights.0,
            embedding_norm_weights.1,
        );

        let mut current_state = &buffer_b;
        let mut intermediate_state = &buffer_a;

        for layer in layers {
            // -- Attention Block --
            encoder.copy_buffer_to_buffer(current_state, 0, &residual_buffer, 0, buffer_size);

            // CORRECT INTEGRATION: Call the full attention block function.
            run_gpu_attention_block(
                &self.context,
                &mut encoder,
                current_state,
                intermediate_state,
                &layer.attention_weights,
                batch_size,
                seq_len,
                config.hidden_size(),
                config.num_attention_heads(),
            );

            run_gpu_add(
                &self.context,
                &mut encoder,
                intermediate_state,
                &residual_buffer,
                current_state,
                (buffer_size / 4) as u32,
            );
            run_gpu_layer_norm(
                &self.context,
                &mut encoder,
                current_state,
                intermediate_state,
                (batch_size * seq_len) as u32,
                hidden_size as u32,
                config.layer_norm_eps(),
                &layer.attention_weights.norm_weight,
                &layer.attention_weights.norm_bias,
            );

            // -- FFN Block --
            encoder.copy_buffer_to_buffer(intermediate_state, 0, &residual_buffer, 0, buffer_size);

            run_gpu_ffn(
                &self.context,
                &mut encoder,
                intermediate_state,
                current_state,
                &layer.ffn_weights,
                (batch_size * seq_len) as u32,
                config.hidden_size() as u32,
                (config.hidden_size() * 4) as u32,
            );

            run_gpu_add(
                &self.context,
                &mut encoder,
                current_state,
                &residual_buffer,
                intermediate_state,
                (buffer_size / 4) as u32,
            );
            run_gpu_layer_norm(
                &self.context,
                &mut encoder,
                intermediate_state,
                current_state,
                (batch_size * seq_len) as u32,
                hidden_size as u32,
                config.layer_norm_eps(),
                &layer.ffn_weights.norm_weight,
                &layer.ffn_weights.norm_bias,
            );
        }

        self.context.queue.submit(std::iter::once(encoder.finish()));
        let final_ndarray = read_buffer_to_ndarray(
            &self.context,
            current_state,
            (batch_size, seq_len, hidden_size),
        )
        .await?;
        Ok(final_ndarray)
    }
}

fn run_gpu_attention_block(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    input: &Buffer,
    output: &Buffer,
    weights: &GpuAttentionWeights,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
) {
    let device = &context.device;
    let head_dim = hidden_size / num_heads;

    // --- 1. Create Temporary Buffers ---
    let qkv_buffer_size = (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    // Buffers for matmul results
    let q_proj = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Q Proj"),
        size: qkv_buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let k_proj = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("K Proj"),
        size: qkv_buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let v_proj = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("V Proj"),
        size: qkv_buffer_size,
        usage,
        mapped_at_creation: false,
    });

    // A reusable temporary buffer for the results of bias additions
    let proj_biased = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Proj Biased Temp"),
        size: qkv_buffer_size,
        usage,
        mapped_at_creation: false,
    });

    let q_permuted = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Q Permuted"),
        size: qkv_buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let k_permuted_t = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("K Permuted T"),
        size: qkv_buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let v_permuted = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("V Permuted"),
        size: qkv_buffer_size,
        usage,
        mapped_at_creation: false,
    });

    let scores_buffer_size =
        (batch_size * num_heads * seq_len * seq_len * std::mem::size_of::<f32>()) as u64;
    let scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scores"),
        size: scores_buffer_size,
        usage,
        mapped_at_creation: false,
    });

    let context_vectors = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Context Vectors"),
        size: qkv_buffer_size,
        usage,
        mapped_at_creation: false,
    });

    // --- 2. Q, K, V Projections with Bias ---
    let m = (batch_size * seq_len) as u32;
    let k = hidden_size as u32;
    let n = hidden_size as u32;

    // Q = Input * Wq + Bq
    run_gpu_matmul(context, encoder, input, &weights.q_weight, &q_proj, m, k, n);
    // CORRECTED: Write biased result to a new buffer `proj_biased`
    run_gpu_add_bias(
        context,
        encoder,
        &q_proj,
        &weights.q_bias,
        &proj_biased,
        (m * n) as u32,
    );
    run_gpu_reshape(
        context,
        encoder,
        &proj_biased,
        &q_permuted,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        false,
    );

    // K = Input * Wk + Bk
    run_gpu_matmul(context, encoder, input, &weights.k_weight, &k_proj, m, k, n);
    run_gpu_add_bias(
        context,
        encoder,
        &k_proj,
        &weights.k_bias,
        &proj_biased,
        (m * n) as u32,
    );
    run_gpu_reshape(
        context,
        encoder,
        &proj_biased,
        &k_permuted_t,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        true,
    );

    // V = Input * Wv + Bv
    run_gpu_matmul(context, encoder, input, &weights.v_weight, &v_proj, m, k, n);
    run_gpu_add_bias(
        context,
        encoder,
        &v_proj,
        &weights.v_bias,
        &proj_biased,
        (m * n) as u32,
    );
    run_gpu_reshape(
        context,
        encoder,
        &proj_biased,
        &v_permuted,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        false,
    );

    // --- 3. Attention Scores: Q @ K^T ---
    run_gpu_matmul(
        context,
        encoder,
        &q_permuted,
        &k_permuted_t,
        &scores,
        (batch_size * num_heads * seq_len) as u32,
        head_dim as u32,
        seq_len as u32,
    );

    // --- 4. Softmax ---
    let scale = 1.0 / (head_dim as f32).sqrt();
    run_gpu_softmax(
        context,
        encoder,
        &scores,
        (batch_size * num_heads * seq_len) as u32,
        seq_len as u32,
        scale,
    );

    // --- 5. Apply Scores to V: Scores @ V ---
    run_gpu_matmul(
        context,
        encoder,
        &scores,
        &v_permuted,
        &context_vectors,
        (batch_size * num_heads * seq_len) as u32,
        seq_len as u32,
        head_dim as u32,
    );

    // --- 6. "Un-reshape" and Output Projection ---
    run_gpu_unreshape(
        context,
        encoder,
        &context_vectors,
        &proj_biased,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
    );

    // For the final projection, we will use `q_proj` as a temporary buffer for the matmul result.
    run_gpu_matmul(
        context,
        encoder,
        &proj_biased,
        &weights.output_weight,
        &q_proj,
        m,
        k,
        k,
    );
    // CORRECTED: The final `add_bias` writes to the true `output` buffer.
    run_gpu_add_bias(
        context,
        encoder,
        &q_proj,
        &weights.output_bias,
        output,
        (m * n) as u32,
    );
}

/// A self-contained helper to run the `add_bias.wgsl` shader.
///
/// This kernel adds a 1D bias vector to a 2D matrix (`output = input + bias`).
/// It is used after matrix multiplication in projection layers.
fn run_gpu_add_bias(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    input: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    size: u32,
) {
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct Uniforms {
        size: u32,
        _padding: [u32; 3],
    }

    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./shaders/add_bias.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Add Bias Bind Group Layout"),
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
            // @binding(1) Input
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
            // @binding(2) Bias
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
            // @binding(3) Output
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
        label: Some("Add Bias Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Add Bias Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let uniforms = Uniforms {
        size,
        _padding: [0; 3],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Add Bias Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Add Bias Bind Group"),
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
                resource: bias.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Add Bias Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    let workgroup_x = (size + 255) / 256;
    compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
}

/// A self-contained helper to run the batch-aware `reshape.wgsl` shader.
///
/// This kernel is crucial for multi-head attention, permuting the dimensions of the
/// Q, K, and V projections to prepare them for batched matrix multiplication. It's the
/// GPU equivalent of `permute()` and is fully aware of the batch dimension.
fn run_gpu_reshape(
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
    let shader = device.create_shader_module(include_wgsl!("./shaders/reshape.wgsl"));

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

/// A self-contained helper to run the `softmax.wgsl` shader.
///
/// This kernel performs a numerically-stable softmax operation in-place on a buffer.
/// It also applies the attention scaling factor.
fn run_gpu_softmax(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    data: &Buffer, // The buffer to operate on in-place
    rows: u32,
    cols: u32,
    scale: f32,
) {
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct SoftmaxUniforms {
        rows: u32,
        cols: u32,
        scale: f32,
        _padding: u32,
    }

    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./shaders/softmax.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Softmax Bind Group Layout"),
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
            // Note: The buffer is read-write, so we declare it as such.
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Softmax Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Softmax Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let uniforms = SoftmaxUniforms {
        rows,
        cols,
        scale,
        _padding: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Softmax Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Softmax Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Softmax Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    // Dispatch one workgroup per row of the scores matrix
    let workgroup_x = (rows + 255) / 256;
    compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
}
/// A self-contained helper function to run the Add GPU compute shader for residual connections.
fn run_gpu_add(
    context: &WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
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
    let shader = device.create_shader_module(include_wgsl!("./shaders/add.wgsl"));
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

    let uniforms = AddUniforms {
        size,
        _padding: [0; 3],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Add Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Now, the BindGroup will be created with a layout that is guaranteed to match.
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Add Bind Group"),
        layout: &bind_group_layout, // Use the same layout
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
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    let workgroup_x = (size + 255) / 256;
    compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
}

fn run_gpu_layer_norm(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    input: &Buffer,
    output: &Buffer,
    rows: u32,
    cols: u32,
    eps: f32,
    gamma: &Buffer,
    beta: &Buffer,
) {
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct NormUniforms {
        m: u32,
        n: u32,
        eps: f32,
        _padding: u32,
    }

    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./shaders/layer_norm.wgsl"));

    // --- CORRECTED: EXPLICITLY DEFINE THE LAYOUT FOR LAYERNORM ---
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("LayerNorm Bind Group Layout"),
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
        label: Some("LayerNorm Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("LayerNorm Pipeline"),
        layout: Some(&pipeline_layout), // USE THE EXPLICIT LAYOUT
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let uniforms = NormUniforms {
        m: rows,
        n: cols,
        eps,
        _padding: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("LayerNorm Uniform Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("LayerNorm Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
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
                resource: gamma.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: beta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("LayerNorm Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    let workgroup_x = (rows + 255) / 256;
    compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
}
/// A helper function to read a WGPU buffer back to an ndarray::Array3.
async fn read_buffer_to_ndarray(
    context: &WgpuContext,
    buffer: &wgpu::Buffer,
    dims: (usize, usize, usize),
) -> Result<Array3<f32>> {
    let (batch, seq, hidden) = dims;
    let buffer_size = (batch * seq * hidden * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    context.device.poll(wgpu::PollType::wait_indefinitely());

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let result_array = Array3::from_shape_vec((batch, seq, hidden), result_vec)?;
        Ok(result_array)
    } else {
        anyhow::bail!("Failed to read back final result from GPU")
    }
}

fn run_gpu_ffn(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    input: &Buffer,
    output: &Buffer,
    weights: &GpuFeedForwardWeights,
    rows: u32,              // batch_size * seq_len
    hidden_size: u32,       // K
    intermediate_size: u32, // N
) {
    /// A uniform struct to pass metadata (dimensions) to the ffn.wgsl shader.
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct FfnUniforms {
        m: u32,
        k: u32,
        n: u32,
        _padding: u32, // Structs in WGSL are aligned to 16 bytes
    }

    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./shaders/ffn.wgsl"));

    // Define the layout explicitly to match the shader's expectations.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FFN Bind Group Layout"),
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
            // @binding(1) var<storage, read> weights: array<f32>;
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
            // @binding(2) var<storage, read> input: array<f32>;
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
            // @binding(3) var<storage, read_write> output: array<f32>;
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
        label: Some("FFN Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FFN Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FFN Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weights.packed_weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("FFN Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    // The shader processes one row (token) per thread.
    let workgroup_x = (rows + 255) / 256;
    compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
}

/// A generic helper function to run a 2D matrix multiplication `C = A * B` on the GPU.
///
/// This function is the core building block for all dense layers. It takes GPU buffers
/// as input and records the dispatch command into a provided command encoder. It does
/// not perform any data transfers itself.
fn run_gpu_matmul(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    // Input buffers
    a: &Buffer,
    b: &Buffer,
    // Output buffer
    c: &Buffer,
    // Matrix dimensions
    m: u32,
    k: u32,
    n: u32,
) {
    /// A uniform struct to pass matrix dimensions to the shader.
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct MatmulInfo {
        m: u32,
        k: u32,
        n: u32,
        _padding: u32,
    }

    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./shaders/matmul_tiled.wgsl"));

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

    let uniforms = MatmulInfo {
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

    // --- THIS IS THE FIX ---
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Matmul Bind Group"),
        layout: &bind_group_layout, // CORRECTED: Use the `bind_group_layout` variable
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
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    const TILE_DIM: u32 = 16;
    let workgroup_x = (n + TILE_DIM - 1) / TILE_DIM;
    let workgroup_y = (m + TILE_DIM - 1) / TILE_DIM;

    compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
}
/// A self-contained helper to run the `unreshape.wgsl` shader.
///
/// This kernel is the inverse of the reshape operation, taking the context vectors
/// from the multi-head format `[B, H, S, D]` and permuting them back into the
/// standard sequence format `[B, S, H*D]`, ready for the final output projection.
fn run_gpu_unreshape(
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
    let shader = device.create_shader_module(include_wgsl!("./shaders/unreshape.wgsl"));

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
mod tests {
    use super::*;
    use crate::{LayerNorm, wgpu_context::WgpuContext, FeedForward};
    use ndarray::{Array, Array1, Array3, Array2};
    use ndarray_rand::RandomExt;
    use rand_distr::Uniform; // <--- this is the correct one

    /// A helper function to get a WGPU context for testing.
    /// Panics if a GPU adapter cannot be found.
    async fn get_test_context() -> Arc<WgpuContext> {
        Arc::new(WgpuContext::new().await)
    }

    /// A crucial helper for comparing floating-point vectors for near-equality.
    /// Direct comparison `assert_eq!` will fail due to tiny precision differences
    /// between CPU and GPU floating-point math.
    fn assert_vecs_are_close(vec1: &[f32], vec2: &[f32], tolerance: f32) {
        assert_eq!(vec1.len(), vec2.len(), "Vectors have different lengths");
        for (i, (a, b)) in vec1.iter().zip(vec2.iter()).enumerate() {
            if (a - b).abs() > tolerance {
                panic!(
                    "Mismatch at index {}: cpu = {}, gpu = {}. Difference: {}",
                    i,
                    a,
                    b,
                    (a - b).abs()
                );
            }
        }
    }

    /// Tests that the `run_gpu_layer_norm` kernel produces the same result as the CPU `LayerNorm`.
    #[tokio::test]
    async fn test_layer_norm_correctness() -> Result<()> {
        let context = get_test_context().await;
        let device = &context.device;

        // --- 1. Arrange: Create Identical Inputs for CPU and GPU ---

        let batch_size = 1;
        let seq_len = 8;
        let hidden_size = 32;
        let eps = 1e-5;

        // Create random input data
        let input_cpu: Array3<f32> =
            Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
        let gamma_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(0.5, 1.5));
        let beta_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

        // Upload the same data to GPU buffers
        let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Input"),
            contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let gamma_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Gamma"),
            contents: bytemuck::cast_slice(gamma_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let beta_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Beta"),
            contents: bytemuck::cast_slice(beta_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Output"),
            size: (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // --- 2. Act: Run both CPU and GPU versions ---

        // Get the ground truth from the CPU implementation
        let cpu_layernorm = LayerNorm::new(gamma_cpu, beta_cpu, eps);
        let cpu_result = cpu_layernorm.forward_3d(&input_cpu);

        // Run the GPU kernel
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Encoder"),
        });
        run_gpu_layer_norm(
            &context,
            &mut encoder,
            &input_gpu,
            &output_gpu,
            (batch_size * seq_len) as u32,
            hidden_size as u32,
            eps,
            &gamma_gpu,
            &beta_gpu,
        );
        context.queue.submit(std::iter::once(encoder.finish()));

        // Read the result back from the GPU
        let gpu_result_array =
            read_buffer_to_ndarray(&context, &output_gpu, (batch_size, seq_len, hidden_size))
                .await?;

        // --- 3. Assert: Compare the results ---

        println!("Verifying LayerNorm GPU kernel against CPU implementation...");
        assert_vecs_are_close(
            cpu_result.as_slice().unwrap(),
            gpu_result_array.as_slice().unwrap(),
            1e-4, // A reasonable tolerance for f32 differences between CPU/GPU
        );
        println!("✅ LayerNorm GPU implementation is correct!");

        Ok(())
    }
    /// Tests that the `run_gpu_add` kernel produces the same result as CPU `+`.
    #[tokio::test]
    async fn test_add_correctness() -> Result<()> {
        let context = get_test_context().await;
        let device = &context.device;

        // --- 1. Arrange ---
        let batch_size = 2;
        let seq_len = 16;
        let hidden_size = 64;
        let total_elements = (batch_size * seq_len * hidden_size) as u32;

        let input_a_cpu: Array3<f32> = Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
        let input_b_cpu: Array3<f32> = Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));

        let input_a_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Add Input A"),
            contents: bytemuck::cast_slice(input_a_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let input_b_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Add Input B"),
            contents: bytemuck::cast_slice(input_b_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Add Output"),
            size: (total_elements as u64) * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // --- 2. Act ---
        let cpu_result = &input_a_cpu + &input_b_cpu;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        run_gpu_add(&context, &mut encoder, &input_a_gpu, &input_b_gpu, &output_gpu, total_elements);
        context.queue.submit(std::iter::once(encoder.finish()));
        
        let gpu_result_array = read_buffer_to_ndarray(&context, &output_gpu, (batch_size, seq_len, hidden_size)).await?;
        
        // --- 3. Assert ---
        println!("Verifying Add GPU kernel against CPU implementation...");
        assert_vecs_are_close(cpu_result.as_slice().unwrap(), gpu_result_array.as_slice().unwrap(), 1e-6);
        println!("✅ Add GPU implementation is correct!");

        Ok(())
    }

    /// Tests that the `run_gpu_ffn` kernel produces the same result as the CPU `FeedForward`.
     #[tokio::test]
    async fn test_ffn_correctness() -> Result<()> {
        let context = get_test_context().await;
        let device = &context.device;

        // --- 1. Arrange ---
        let batch_size = 1;
        let seq_len = 8;
        let hidden_size = 32;
        let intermediate_size = hidden_size * 4;

        let input_cpu: Array3<f32> = Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
        let intermediate_w_cpu: Array2<f32> = Array::random((hidden_size, intermediate_size), Uniform::new(-0.5, 0.5));
        let intermediate_b_cpu: Array1<f32> = Array::random(intermediate_size, Uniform::new(-0.5, 0.5));
        let output_w_cpu: Array2<f32> = Array::random((intermediate_size, hidden_size), Uniform::new(-0.5, 0.5));
        let output_b_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

        // --- THIS IS THE FIX ---
        // We must ensure the transposed array has a standard memory layout before calling .as_slice().
        let intermediate_w_t = intermediate_w_cpu.t().as_standard_layout().to_owned();
        let output_w_t = output_w_cpu.t().as_standard_layout().to_owned();
        
        // Now, the `.as_slice().unwrap()` calls are guaranteed to succeed.
        let mut packed_ffn_data: Vec<f32> = Vec::new();
        packed_ffn_data.extend_from_slice(intermediate_w_t.as_slice().unwrap());
        packed_ffn_data.extend_from_slice(intermediate_b_cpu.as_slice().unwrap());
        packed_ffn_data.extend_from_slice(output_w_t.as_slice().unwrap());
        packed_ffn_data.extend_from_slice(output_b_cpu.as_slice().unwrap());

        // --- Upload GPU data (no changes here) ---
        let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test FFN Input"),
            contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let packed_weights_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test FFN Packed Weights"),
            contents: bytemuck::cast_slice(&packed_ffn_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test FFN Output"),
            size: (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create dummy GpuFeedForwardWeights for the test. The norm weights are not used
        // by `run_gpu_ffn`, but the struct requires them.
        let dummy_norm_w_cpu: Array1<f32> = Array1::zeros(hidden_size);
        let dummy_norm_b_cpu: Array1<f32> = Array1::zeros(hidden_size);
        
        let dummy_norm_w_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Norm W"),
            contents: bytemuck::cast_slice(dummy_norm_w_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_norm_b_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Norm B"),
            contents: bytemuck::cast_slice(dummy_norm_b_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let gpu_weights = GpuFeedForwardWeights {
            packed_weights: Arc::new(packed_weights_gpu),
            norm_weight: Arc::new(dummy_norm_w_gpu),
            norm_bias: Arc::new(dummy_norm_b_gpu),
        };

        // --- 2. Act ---
        let cpu_ffn = FeedForward::new(
            intermediate_w_t, intermediate_b_cpu.clone(),
            output_w_t, output_b_cpu.clone()
        );
        let cpu_result = cpu_ffn.forward(&input_cpu)?;
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FFN Test Encoder") });
        run_gpu_ffn(
            &context, &mut encoder, &input_gpu, &output_gpu, &gpu_weights,
            (batch_size * seq_len) as u32, hidden_size as u32, intermediate_size as u32
        );
        context.queue.submit(std::iter::once(encoder.finish()));

        let gpu_result_array = read_buffer_to_ndarray(&context, &output_gpu, (batch_size, seq_len, hidden_size)).await?;

        // --- 3. Assert ---
        println!("Verifying FFN GPU kernel against CPU implementation...");
        assert_vecs_are_close(cpu_result.as_slice().unwrap(), gpu_result_array.as_slice().unwrap(), 1e-4);
        println!("✅ FFN GPU implementation is correct!");

        Ok(())
    }
}

