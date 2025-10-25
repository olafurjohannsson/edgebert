//! A reusable orchestrator for running a generic transformer encoder pipeline on the GPU.

use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, s};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::current;
use std::time::Instant;
use wgpu::ComputePipeline;
use wgpu::util::DeviceExt;
use wgpu::wgc::pipeline;
use wgpu::{Buffer, CommandEncoder, PollType, include_wgsl};

use crate::gpu_ops::add::compile_add_pipeline;
use crate::gpu_ops::add_bias::compile_add_bias_pipeline;
use crate::gpu_ops::apply_mask::compile_apply_mask_pipeline;
use crate::gpu_ops::ffn::{compile_fc1_pipeline, compile_fc2_pipeline};

//compile_ffn_pipeline;
use crate::gpu_ops::layer_norm::compile_layer_norm_pipeline;
use crate::gpu_ops::matmul::{compile_bmm_pipeline, compile_matmul_pipeline};
use crate::gpu_ops::reshape::{compile_reshape_pipeline, compile_unreshape_pipeline};
use crate::gpu_ops::softmax::compile_softmax_pipeline;

use crate::bind_group::{BindGroupCache, CacheStats};

use crate::MultiHeadAttention;
use crate::gpu_ops::{
    add::run_gpu_add,
    add_bias::run_gpu_add_bias,
    apply_mask::run_gpu_apply_mask,
    common::read_buffer_to_ndarray,
    ffn::{GpuFeedForwardWeights, run_gpu_ffn},
    layer_norm::run_gpu_layer_norm,
    matmul::{run_gpu_bmm, run_gpu_matmul},
    reshape::{run_gpu_reshape, run_gpu_unreshape},
    softmax::run_gpu_softmax,
};
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

/// A container for all temporary, reusable buffers required by the `run_gpu_attention_block`.
///
/// Creating these buffers is expensive, so they are created once per `forward` pass
/// and then reused by each layer, dramatically reducing overhead.
struct AttentionTempBuffers {
    // Buffers for the initial Q, K, V projections after matmul
    q_proj: Buffer,
    k_proj: Buffer,
    v_proj: Buffer,
    // A reusable buffer for the output of bias additions
    proj_biased: Buffer,
    // Buffers for the reshaped Q, K, V tensors
    q_permuted: Buffer,
    k_permuted_t: Buffer,
    v_permuted: Buffer,
    // Buffer for the raw attention scores (Q @ K^T)
    scores: Buffer,
    // Buffer for the context vectors (Softmax(Scores) @ V)
    context_vectors: Buffer,
    pub ffn_intermediate: Buffer,
}

#[derive(Clone)]
pub struct GpuTransformerLayer {
    pub attention_weights: GpuAttentionWeights,
    pub ffn_weights: GpuFeedForwardWeights,
}

/// The reusable orchestrator for the GPU encoder pipeline.
pub struct GpuEncoderPipeline {
    context: Arc<WgpuContext>,
    pipeline: HashMap<Pipeline, Arc<ComputePipeline>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pipeline {
    // todo: change to PipelineType
    Add,
    AddBias,
    ApplyMask,
    FFN,
    FC1,
    FC2,
    LayerNorm,
    BMM,
    MatMul,
    Reshape,
    Unreshape,
    Softmax,
}

impl GpuEncoderPipeline {
    pub fn new(context: Arc<WgpuContext>) -> Result<Self> {
        let mut pipeline = HashMap::new();
        println!("Compiling GPU compute shaders...");

        let device = &context.device;

        let add_pipeline = Arc::new(compile_add_pipeline(&context));
        pipeline.insert(Pipeline::Add, add_pipeline);

        let add_bias_pipeline = Arc::new(compile_add_bias_pipeline(&context));
        pipeline.insert(Pipeline::AddBias, add_bias_pipeline);

        let apply_mask_pipeline = Arc::new(compile_apply_mask_pipeline(&context));
        pipeline.insert(Pipeline::ApplyMask, apply_mask_pipeline);

        // let ffn_pipeline = Arc::new(compile_ffn_pipeline(&context));

        let fc1_pipeline = Arc::new(compile_fc1_pipeline(&context));
        pipeline.insert(Pipeline::FC1, fc1_pipeline);

        let fc2_pipeline = Arc::new(compile_fc2_pipeline(&context));
        pipeline.insert(Pipeline::FC2, fc2_pipeline);

        let layer_norm_pipeline = Arc::new(compile_layer_norm_pipeline(&context));
        pipeline.insert(Pipeline::LayerNorm, layer_norm_pipeline);

        let bmm_pipeline = Arc::new(compile_bmm_pipeline(&context));
        pipeline.insert(Pipeline::BMM, bmm_pipeline);

        let matmul_pipeline = Arc::new(compile_matmul_pipeline(&context));

        pipeline.insert(Pipeline::MatMul, matmul_pipeline);

        let reshape_pipeline = Arc::new(compile_reshape_pipeline(&context));
        pipeline.insert(Pipeline::Reshape, reshape_pipeline);

        let unreshape_pipeline = Arc::new(compile_unreshape_pipeline(&context));
        pipeline.insert(Pipeline::Unreshape, unreshape_pipeline);

        let softmax_pipeline = Arc::new(compile_softmax_pipeline(&context));
        pipeline.insert(Pipeline::Softmax, softmax_pipeline);

        Ok(Self {
            context,
            pipeline,
        })
    }
    fn create_aligned_buffer(
        &self,
        device: &wgpu::Device,
        data: &[f32],
        label: &str,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let mut aligned_data = data.to_vec();

        // Ensure size is multiple of 4 (16 bytes)
        let remainder = aligned_data.len() % 4;
        if remainder != 0 {
            aligned_data.resize(aligned_data.len() + (4 - remainder), 0.0);
        }

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&aligned_data),
            usage,
        })
    }
    /// Executes the full, end-to-end GPU forward pass for a transformer encoder.
    pub async fn forward<C: TransformerConfig + ?Sized>(
        &self,
        config: &C,
        initial_embeddings: &Array3<f32>,
        attention_mask: &Array2<f32>,
        embedding_norm_weights: (&Buffer, &Buffer),
        layers: &[GpuTransformerLayer],
    ) -> Result<Array3<f32>> {
        use std::time::Instant;
        let total_start = Instant::now();

        let (batch_size, seq_len, hidden_size) = initial_embeddings.dim();
        println!("\n=== GPU Forward Pass ===");
        println!(
            "Batch: {}, SeqLen: {}, Hidden: {}",
            batch_size, seq_len, hidden_size
        );

        let device = &self.context.device;

        // Create timestamp query set for detailed profiling
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Timing"),
            ty: wgpu::QueryType::Timestamp,
            count: 100, // Enough for many operations
        });

        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Buffer"),
            size: 100 * 8, // 50 timestamps × 8 bytes
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let query_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Readback"),
            size: 100 * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_size = (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>())
            as wgpu::BufferAddress;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        // === BUFFER CREATION ===
        let buffer_start = Instant::now();
        let buffer_a = self.create_aligned_buffer(
            device,
            initial_embeddings.as_standard_layout().as_slice().unwrap(),
            "Pipeline Buffer A",
            usage,
        );

        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline Buffer B"),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });

        let residual_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Residual Buffer"),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });

        let mask_buffer = self.create_aligned_buffer(
            device,
            attention_mask.as_slice().unwrap(),
            "Attention Mask Buffer",
            wgpu::BufferUsages::STORAGE,
        );
        println!("Main buffers created: {:?}", buffer_start.elapsed());

        let encoder_start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder Pipeline"),
        });
        println!("Command encoder created: {:?}", encoder_start.elapsed());

        // === TEMP BUFFERS ===
        let temp_start = Instant::now();
        let temp_buffers = {
            let qkv_buffer_size = buffer_size as u64;
            let scores_buffer_size = (batch_size
                * config.num_attention_heads()
                * seq_len
                * seq_len
                * std::mem::size_of::<f32>()) as u64;
            let ffn_intermediate_size = (batch_size 
                * seq_len 
                * config.hidden_size() * 4  // intermediate_size
                * std::mem::size_of::<f32>()) as u64;

            AttentionTempBuffers {
                q_proj: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Q Proj"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                k_proj: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("K Proj"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                v_proj: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("V Proj"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                proj_biased: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Proj Biased Temp"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                q_permuted: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Q Permuted"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                k_permuted_t: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("K Permuted T"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                v_permuted: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("V Permuted"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                scores: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Scores"),
                    size: scores_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                context_vectors: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Context Vectors"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                ffn_intermediate: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("FFN Intermediate"),
                    size: ffn_intermediate_size,
                    usage,
                    mapped_at_creation: false,
                }),
            }
        };
        println!(
            "Temp buffers (9 buffers) created: {:?}",
            temp_start.elapsed()
        );

        // === GPU ENCODING WITH TIMESTAMPS ===
        let encode_start = Instant::now();
        
        let result_buffer = {
            let cache_start = Instant::now();
            println!("Mutex lock acquired: {:?}", cache_start.elapsed());

            let mut query_idx = 0u32;

            // Start timestamp
            encoder.write_timestamp(&query_set, query_idx);
            query_idx += 1;

            // Initial LayerNorm
            let norm_start = Instant::now();
            run_gpu_layer_norm(
                &self.context,
                &mut encoder,
                &self.pipeline.get(&Pipeline::LayerNorm).unwrap(),
                &buffer_a,
                &buffer_b,
                (batch_size * seq_len) as u32,
                hidden_size as u32,
                config.layer_norm_eps(),
                embedding_norm_weights.0,
                embedding_norm_weights.1,
            );
            encoder.write_timestamp(&query_set, query_idx);
            query_idx += 1;
            println!("Initial LayerNorm encoded: {:?}", norm_start.elapsed());


            let current_state = &buffer_b;
            let intermediate_state = &buffer_a;

            println!("Processing {} layers...", layers.len());
            for (idx, layer) in layers.iter().enumerate() {
                let layer_start = Instant::now();

                // -- Attention Block --
                encoder.copy_buffer_to_buffer(current_state, 0, &residual_buffer, 0, buffer_size);

                // Timestamp before attention
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                run_gpu_attention_block(
                    &self.context,
                    &mut encoder,
                    &self.pipeline,
                    current_state,
                    intermediate_state,
                    &layer.attention_weights,
                    &mask_buffer,
                    &temp_buffers,
                    batch_size,
                    seq_len,
                    config.hidden_size(),
                    config.num_attention_heads(),
                );

                // Timestamp after attention
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                run_gpu_add(
                    &self.context,
                    &mut encoder,
                    &self.pipeline.get(&Pipeline::Add).unwrap(),
                    intermediate_state,
                    &residual_buffer,
                    current_state,
                    (buffer_size / 4) as u32,
                );

                // Timestamp after add
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                run_gpu_layer_norm(
                    &self.context,
                    &mut encoder,
                    &self.pipeline.get(&Pipeline::LayerNorm).unwrap(),
                    current_state,
                    intermediate_state,
                    (batch_size * seq_len) as u32,
                    hidden_size as u32,
                    config.layer_norm_eps(),
                    &layer.attention_weights.norm_weight,
                    &layer.attention_weights.norm_bias,
                );

                // Timestamp after norm
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                // -- FFN Block --
                encoder.copy_buffer_to_buffer(
                    intermediate_state,
                    0,
                    &residual_buffer,
                    0,
                    buffer_size,
                );

                // Timestamp before FFN
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                run_gpu_ffn(
                    &self.context,
                    &mut encoder,
                    &self.pipeline.get(&Pipeline::FC1).unwrap(),
                    &self.pipeline.get(&Pipeline::FC2).unwrap(),
                    &temp_buffers.ffn_intermediate,
                    intermediate_state,
                    current_state,
                    &layer.ffn_weights,
                    (batch_size * seq_len) as u32,
                    config.hidden_size() as u32,
                    (config.hidden_size() * 4) as u32,
                );

                // Timestamp after FFN
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                run_gpu_add(
                    &self.context,
                    &mut encoder,
                    &self.pipeline.get(&Pipeline::Add).unwrap(),
                    current_state,
                    &residual_buffer,
                    intermediate_state,
                    (buffer_size / 4) as u32,
                );

                // Timestamp after add
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                run_gpu_layer_norm(
                    &self.context,
                    &mut encoder,
                    &self.pipeline.get(&Pipeline::LayerNorm).unwrap(),
                    intermediate_state,
                    current_state,
                    (batch_size * seq_len) as u32,
                    hidden_size as u32,
                    config.layer_norm_eps(),
                    &layer.ffn_weights.norm_weight,
                    &layer.ffn_weights.norm_bias,
                );

                // Timestamp after final norm
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                println!("  Layer {} encoded: {:?}", idx, layer_start.elapsed());
                    // DEBUG: Read back after first layer

            }

            // Final timestamp
            encoder.write_timestamp(&query_set, query_idx);
            let final_query_idx = query_idx;

            // Resolve all timestamps
            encoder.resolve_query_set(&query_set, 0..final_query_idx + 1, &query_buffer, 0);
            encoder.copy_buffer_to_buffer(
                &query_buffer,
                0,
                &query_readback,
                0,
                (final_query_idx + 1) as u64 * 8,
            );

            let submit_start = Instant::now();
            self.context.queue.submit(std::iter::once(encoder.finish()));
            println!("Commands submitted to GPU: {:?}", submit_start.elapsed());

            current_state.clone()
        };
        println!("Total encoding time: {:?}", encode_start.elapsed());

        // === WAIT FOR GPU ===
        let gpu_wait_start = Instant::now();
        self.context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        println!("GPU completion wait: {:?}", gpu_wait_start.elapsed());

        // === READ AND ANALYZE TIMESTAMPS ===
        let timestamp_start = Instant::now();
        let slice = query_readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely());

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        println!("\n=== GPU OPERATION BREAKDOWN ===");
        let total_gpu_time =
            (timestamps[timestamps.len() - 1] - timestamps[0]) as f64 / 1_000_000.0;
        println!("Total GPU time: {:.2}ms", total_gpu_time);

        let initial_layernorm = (timestamps[1] - timestamps[0]) as f64 / 1_000_000.0;
        println!("Initial LayerNorm: {:.2}ms", initial_layernorm);

        let mut idx = 2;
        let mut total_attn = 0.0;
        let mut total_ffn = 0.0;
        let mut total_adds = 0.0;
        let mut total_norms = 0.0;

        for layer_num in 0..layers.len() {
            let attn_time = (timestamps[idx + 1] - timestamps[idx]) as f64 / 1_000_000.0;
            let attn_add = (timestamps[idx + 2] - timestamps[idx + 1]) as f64 / 1_000_000.0;
            let attn_norm = (timestamps[idx + 3] - timestamps[idx + 2]) as f64 / 1_000_000.0;
            let ffn_time = (timestamps[idx + 5] - timestamps[idx + 4]) as f64 / 1_000_000.0;
            let ffn_add = (timestamps[idx + 6] - timestamps[idx + 5]) as f64 / 1_000_000.0;
            let ffn_norm = (timestamps[idx + 7] - timestamps[idx + 6]) as f64 / 1_000_000.0;

            total_attn += attn_time;
            total_ffn += ffn_time;
            total_adds += attn_add + ffn_add;
            total_norms += attn_norm + ffn_norm;

            println!(
                "Layer {}: Attn={:.2}ms, AttnAdd={:.2}ms, AttnNorm={:.2}ms, FFN={:.2}ms, FFNAdd={:.2}ms, FFNNorm={:.2}ms",
                layer_num, attn_time, attn_add, attn_norm, ffn_time, ffn_add, ffn_norm
            );

            idx += 8;
        }

        println!("\n=== TOTALS ===");
        println!(
            "Total Attention: {:.2}ms ({:.1}%)",
            total_attn,
            100.0 * total_attn / total_gpu_time
        );
        println!(
            "Total FFN: {:.2}ms ({:.1}%)",
            total_ffn,
            100.0 * total_ffn / total_gpu_time
        );
        println!(
            "Total Adds: {:.2}ms ({:.1}%)",
            total_adds,
            100.0 * total_adds / total_gpu_time
        );
        println!(
            "Total Norms: {:.2}ms ({:.1}%)",
            total_norms,
            100.0 * total_norms / total_gpu_time
        );

        drop(data);
        query_readback.unmap();
        println!("Timestamp analysis: {:?}", timestamp_start.elapsed());

        // === READBACK RESULT ===
        let read_start = Instant::now();
        let final_ndarray = read_buffer_to_ndarray(
            &self.context,
            &result_buffer,
            (batch_size, seq_len, hidden_size),
        )
        .await?;
        println!("Result readback only: {:?}", read_start.elapsed());

        println!("=== TOTAL FORWARD: {:?} ===\n", total_start.elapsed());
        Ok(final_ndarray)
    }
}

fn run_gpu_attention_block(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &HashMap<Pipeline, Arc<ComputePipeline>>,
    input: &Buffer,
    output: &Buffer,
    weights: &GpuAttentionWeights,
    mask: &Buffer,
    temp: &AttentionTempBuffers,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
) {
    // use std::time::Instant;

    // let device = &context.device;
    let head_dim = hidden_size / num_heads;

    let m = (batch_size * seq_len) as u32;
    let k = hidden_size as u32;

    // Q = Input * Wq + Bq
    // Q Path: input -> q_proj -> proj_biased -> q_permuted
    // let start = Instant::now();
    run_gpu_matmul(
        context,
        encoder,
        pipeline.get(&Pipeline::MatMul).unwrap(),
        input,
        &weights.q_weight,
        &temp.q_proj,
        m,
        k,
        k,
    );
    // println!("      Q matmul: {:?}", start.elapsed());

    // let start = Instant::now();
    run_gpu_add_bias(
        context,
        encoder,
        pipeline.get(&Pipeline::AddBias).unwrap(),
        &temp.q_proj,
        &weights.q_bias,
        &temp.proj_biased,
        m * k,
    );
    // println!("      Q bias: {:?}", start.elapsed());

    // let start = Instant::now();
    run_gpu_reshape(
        context,
        encoder,
        pipeline.get(&Pipeline::Reshape).unwrap(),
        &temp.proj_biased,
        &temp.q_permuted,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        false,
    );
    // println!("      Q reshape: {:?}", start.elapsed());

    // // K Path: input -> k_proj -> proj_biased -> k_permuted_t
    // let start = Instant::now();
    run_gpu_matmul(
        context,
        encoder,
        pipeline.get(&Pipeline::MatMul).unwrap(),
        input,
        &weights.k_weight,
        &temp.k_proj,
        m,
        k,
        k,
    );
    // println!("      K matmul: {:?}", start.elapsed());

    // let start = Instant::now();
    run_gpu_add_bias(
        context,
        encoder,
        pipeline.get(&Pipeline::AddBias).unwrap(),
        &temp.k_proj,
        &weights.k_bias,
        &temp.proj_biased,
        m * k,
    );
    // println!("      K bias: {:?}", start.elapsed());

    // let start = Instant::now();
    run_gpu_reshape(
        context,
        encoder,
        pipeline.get(&Pipeline::Reshape).unwrap(),
        &temp.proj_biased,
        &temp.k_permuted_t,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        true,
    );
    // println!("      K reshape: {:?}", start.elapsed());

    // // V Path: input -> v_proj -> proj_biased -> v_permuted
    // let start = Instant::now();
    run_gpu_matmul(
        context,
        encoder,
        pipeline.get(&Pipeline::MatMul).unwrap(),
        input,
        &weights.v_weight,
        &temp.v_proj,
        m,
        k,
        k,
    );
    // println!("      V matmul: {:?}", start.elapsed());

    // let start = Instant::now();
    run_gpu_add_bias(
        context,
        encoder,
        pipeline.get(&Pipeline::AddBias).unwrap(),
        &temp.v_proj,
        &weights.v_bias,
        &temp.proj_biased,
        m * k,
    );
    // println!("      V bias: {:?}", start.elapsed());

    // let start = Instant::now();
    run_gpu_reshape(
        context,
        encoder,
        pipeline.get(&Pipeline::Reshape).unwrap(),
        &temp.proj_biased,
        &temp.v_permuted,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        false,
    );
    // println!("      V reshape: {:?}", start.elapsed());

    // // --- 3. Attention Scores: Q @ K^T ---
    // let start = Instant::now();
    run_gpu_bmm(
        context,
        encoder,
        pipeline.get(&Pipeline::BMM).unwrap(),
        &temp.q_permuted,
        &temp.k_permuted_t,
        &temp.scores,
        (batch_size * num_heads) as u32,
        seq_len as u32,
        head_dim as u32,
        seq_len as u32,
    );
    // println!("      Scores BMM (Q@K^T): {:?}", start.elapsed());

    // // --- 4. Apply Mask ---
    // let start = Instant::now();
    run_gpu_apply_mask(
        context,
        encoder,
        pipeline.get(&Pipeline::ApplyMask).unwrap(),
        &temp.scores,
        mask,
        batch_size as u32,
        num_heads as u32,
        seq_len as u32,
    );
    // println!("      Apply mask: {:?}", start.elapsed());

    // // --- 5. Softmax (in-place on scores) ---
    // let start = Instant::now();
    let scale = 1.0 / (head_dim as f32).sqrt();
    run_gpu_softmax(
        context,
        encoder,
        pipeline.get(&Pipeline::Softmax).unwrap(),
        &temp.scores,
        (batch_size * num_heads * seq_len) as u32,
        seq_len as u32,
        scale,
    );
    // println!("      Softmax: {:?}", start.elapsed());

    // // --- 6. Apply Scores to V: Scores @ V ---
    // let start = Instant::now();
    run_gpu_bmm(
        context,
        encoder,
        pipeline.get(&Pipeline::BMM).unwrap(),
        &temp.scores,
        &temp.v_permuted,
        &temp.context_vectors,
        (batch_size * num_heads) as u32,
        seq_len as u32,
        seq_len as u32,
        head_dim as u32,
    );
    // println!("      Context BMM (Scores@V): {:?}", start.elapsed());

    // // --- 7. "Un-reshape" and Output Projection ---
    // let start = Instant::now();
    run_gpu_unreshape(
        context,
        encoder,
        pipeline.get(&Pipeline::Unreshape).unwrap(),
        &temp.context_vectors,
        &temp.proj_biased,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
    );
    // println!("      Unreshape: {:?}", start.elapsed());

    // // Use `temp.q_proj` as a temporary buffer for the matmul result before the final bias add.
    // let start = Instant::now();
    run_gpu_matmul(
        context,
        encoder,
        pipeline.get(&Pipeline::MatMul).unwrap(),
        &temp.proj_biased,
        &weights.output_weight,
        &temp.q_proj,
        m,
        k,
        k,
    );
    // println!("      Output matmul: {:?}", start.elapsed());

    // // The final result is written to the main `output` buffer.
    // let start = Instant::now();
    run_gpu_add_bias(
        context,
        encoder,
        pipeline.get(&Pipeline::AddBias).unwrap(),
        &temp.q_proj,
        &weights.output_bias,
        output,
        m * k,
    );
    // println!("      Output bias: {:?}", start.elapsed());
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeedForward, LayerNorm, gpu_ops::softmax, wgpu_context::WgpuContext};
    use ndarray::{Array, Array1, Array2, Array3};
    use ndarray_rand::RandomExt;
    use rand_distr::Uniform;
    

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

    #[tokio::test]
    async fn test_attention_q_projection_correctness() -> Result<()> {
        let context = get_test_context().await;
        let device = &context.device;

        // --- 1. Arrange ---
        let (batch_size, seq_len, hidden_size, num_heads) = (1, 16, 64, 4);
        let head_dim = hidden_size / num_heads;

        // Create random CPU data
        let input_cpu: Array3<f32> =
            Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
        let q_w_cpu: Array2<f32> =
            Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
        let q_b_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

        // --- 2. Act (CPU Path - Ground Truth) ---

        // a) Perform the matrix multiplication: Input @ Wq^T
        // We use `.dot()` on a 2D view of the input.
        let input_2d = input_cpu
            .as_standard_layout()
            .into_shape((seq_len, hidden_size))?;
        let q_proj_cpu = input_2d.dot(&q_w_cpu); // Note: Here we don't transpose, so we compare against a non-transposed GPU weight later

        // b) Add the bias
        let q_biased_cpu = q_proj_cpu + &q_b_cpu;

        // c) Manually reshape [S, H*D] -> [H, S, D] to match the GPU kernel's output layout
        let mut cpu_q_permuted = Array3::<f32>::zeros((num_heads, seq_len, head_dim));
        for s_idx in 0..seq_len {
            for h_idx in 0..num_heads {
                for d_idx in 0..head_dim {
                    let val = q_biased_cpu[[s_idx, h_idx * head_dim + d_idx]];
                    cpu_q_permuted[[h_idx, s_idx, d_idx]] = val;
                }
            }
        }

        // --- 2. Act (GPU Path) ---

        // a) Upload data to GPU
        let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Q Proj Input"),
            contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // IMPORTANT: We upload the NON-TRANSPOSED weight to match the CPU calculation above.
        let q_weight_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Q Proj Weight"),
            contents: bytemuck::cast_slice(q_w_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let q_bias_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Q Proj Bias"),
            contents: bytemuck::cast_slice(q_b_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buffer_size = (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        // Create temporary buffers for the GPU calculation
        let q_proj_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Q Proj Matmul Out"),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let q_biased_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Q Proj Bias Out"),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let q_permuted_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Q Proj Permuted Out"),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });

        // b) Record the GPU commands
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let matmul_pipeline = compile_matmul_pipeline(&context);
        let add_bias_pipeline = compile_add_bias_pipeline(&context);
        let reshape_pipeline = compile_reshape_pipeline(&context);
        
        run_gpu_matmul(
            &context,
            &mut encoder,
            &matmul_pipeline,
            &input_gpu,
            &q_weight_gpu,
            &q_proj_gpu,
            (batch_size * seq_len) as u32,
            hidden_size as u32,
            hidden_size as u32,
        );
        run_gpu_add_bias(
            &context,
            &mut encoder,
            &add_bias_pipeline,
            &q_proj_gpu,
            &q_bias_gpu,
            &q_biased_gpu,
            (batch_size * seq_len * hidden_size) as u32,
        );
        run_gpu_reshape(
            &context,
            &mut encoder,
            &reshape_pipeline,
            &q_biased_gpu,
            &q_permuted_gpu,
            batch_size as u32,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
            false,
        );
        context.queue.submit(std::iter::once(encoder.finish()));

        // c) Read back the final result of this sub-pipeline
        let gpu_q_permuted_array =
            read_buffer_to_ndarray(&context, &q_permuted_gpu, (num_heads, seq_len, head_dim))
                .await?;

        // --- 3. Assert ---
        println!("Verifying Attention Q-Projection GPU kernel against CPU implementation...");
        assert_vecs_are_close(
            cpu_q_permuted.as_slice().unwrap(),
            gpu_q_permuted_array.as_slice().unwrap(),
            1e-4,
        );
        println!("✅ Attention Q-Projection and Reshape are correct!");

        Ok(())
    }

    #[tokio::test]
    async fn test_attention_scores_correctness2() -> Result<()> {
        let context = get_test_context().await;
        let device = &context.device;

        // --- 1. Arrange ---
        let (batch_size, seq_len, hidden_size, num_heads) = (1, 8, 32, 4);
        let head_dim = hidden_size / num_heads;

        let input_cpu: Array3<f32> =
            Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
        let q_w_cpu: Array2<f32> =
            Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
        let q_b_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
        let k_w_cpu: Array2<f32> =
            Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
        let k_b_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

        // --- GPU Buffer Setup ---
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let qkv_buffer_size =
            (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64;

        let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scores Test Input"),
            contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
            usage,
        });
        let q_weight_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scores Test Q Weight"),
            contents: bytemuck::cast_slice(q_w_cpu.as_slice().unwrap()),
            usage,
        });
        let q_bias_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scores Test Q Bias"),
            contents: bytemuck::cast_slice(q_b_cpu.as_slice().unwrap()),
            usage,
        });
        let k_weight_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scores Test K Weight"),
            contents: bytemuck::cast_slice(k_w_cpu.as_slice().unwrap()),
            usage,
        });
        let k_bias_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scores Test K Bias"),
            contents: bytemuck::cast_slice(k_b_cpu.as_slice().unwrap()),
            usage,
        });

        let q_proj = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores Q Proj"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let k_proj = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores K Proj"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let proj_biased = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores Proj Biased"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let q_permuted = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores Q Permuted"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let k_permuted_t = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores K Permuted T"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        });

        let scores_buffer_size =
            (batch_size * num_heads * seq_len * seq_len * std::mem::size_of::<f32>()) as u64;
        let scores_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores Output"),
            size: scores_buffer_size,
            usage,
            mapped_at_creation: false,
        });

        // --- 2. Act & Assert, Step-by-Step ---

        // == Step A: Verify Q-Permuted ==
        let input_2d = input_cpu.view().into_shape((seq_len, hidden_size))?;
        let q_biased_cpu = input_2d.dot(&q_w_cpu) + &q_b_cpu;
        let mut cpu_q_permuted = Array3::<f32>::zeros((num_heads, seq_len, head_dim));
        for s_idx in 0..seq_len {
            for h_idx in 0..num_heads {
                for d_idx in 0..head_dim {
                    let val = q_biased_cpu[[s_idx, h_idx * head_dim + d_idx]];
                    cpu_q_permuted[[h_idx, s_idx, d_idx]] = val;
                }
            }
        }

        let mut encoder_q =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let matmul_pipeline = compile_matmul_pipeline(&context);
        let add_bias_pipeline = compile_add_bias_pipeline(&context);
        let reshape_pipeline = compile_reshape_pipeline(&context);
        
        run_gpu_matmul(
            &context,
            &mut encoder_q,
            &matmul_pipeline,
            &input_gpu,
            &q_weight_gpu,
            &q_proj,
            (seq_len * batch_size) as u32,
            hidden_size as u32,
            hidden_size as u32,
        );
        run_gpu_add_bias(
            &context,
            &mut encoder_q,
            &add_bias_pipeline,
            &q_proj,
            &q_bias_gpu,
            &proj_biased,
            (seq_len * hidden_size * batch_size) as u32,
        );
        run_gpu_reshape(
            &context,
            &mut encoder_q,
            &reshape_pipeline,
            &proj_biased,
            &q_permuted,
            batch_size as u32,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
            false,
        );
        context.queue.submit(std::iter::once(encoder_q.finish()));
        let gpu_q_permuted_array =
            read_buffer_to_ndarray(&context, &q_permuted, (num_heads, seq_len, head_dim)).await?;

        println!("Verifying intermediate Q-Permuted...");
        assert_vecs_are_close(
            cpu_q_permuted.as_slice().unwrap(),
            gpu_q_permuted_array.as_slice().unwrap(),
            1e-4,
        );
        println!("✅ Intermediate Q-Permuted is correct.");

        // == Step B: Verify K-Permuted-Transposed ==
        let k_biased_cpu = input_2d.dot(&k_w_cpu) + &k_b_cpu;
        let mut cpu_k_permuted_t = Array3::<f32>::zeros((num_heads, head_dim, seq_len));
        // Manual reshape for K^T: [S, H*D] -> [H, D, S]
        for s_idx in 0..seq_len {
            for h_idx in 0..num_heads {
                for d_idx in 0..head_dim {
                    let val = k_biased_cpu[[s_idx, h_idx * head_dim + d_idx]];
                    cpu_k_permuted_t[[h_idx, d_idx, s_idx]] = val;
                }
            }
        }

        let mut encoder_k =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        run_gpu_matmul(
            &context,
            &mut encoder_k,
            &matmul_pipeline,
            &input_gpu,
            &k_weight_gpu,
            &k_proj,
            (seq_len * batch_size) as u32,
            hidden_size as u32,
            hidden_size as u32,
        );
        run_gpu_add_bias(
            &context,
            &mut encoder_k,
            &add_bias_pipeline,
            &k_proj,
            &k_bias_gpu,
            &proj_biased,
            (seq_len * hidden_size * batch_size) as u32,
        );
        run_gpu_reshape(
            &context,
            &mut encoder_k,
            &reshape_pipeline,
            &proj_biased,
            &k_permuted_t,
            batch_size as u32,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
            true,
        );
        context.queue.submit(std::iter::once(encoder_k.finish()));
        let gpu_k_permuted_t_array =
            read_buffer_to_ndarray(&context, &k_permuted_t, (num_heads, head_dim, seq_len)).await?;

        println!("Verifying intermediate K-Permuted-Transposed...");
        assert_vecs_are_close(
            cpu_k_permuted_t.as_slice().unwrap(),
            gpu_k_permuted_t_array.as_slice().unwrap(),
            1e-4,
        );
        println!("✅ Intermediate K-Permuted-Transposed is correct.");

        // == Step C: Verify Final Scores ==
        let mut scores_cpu = Array3::<f32>::zeros((num_heads, seq_len, seq_len));
        for i in 0..num_heads {
            let q_head = cpu_q_permuted.slice(s![i, .., ..]);
            let k_head = cpu_k_permuted_t.slice(s![i, .., ..]);
            scores_cpu
                .slice_mut(s![i, .., ..])
                .assign(&q_head.dot(&k_head));
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        scores_cpu *= scale;
        for i in 0..num_heads {
            for j in 0..seq_len {
                let mut row = scores_cpu.slice_mut(s![i, j, ..]);
                let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                row.mapv_inplace(|x| (x - max_val).exp());
                let sum = row.sum();
                if sum > 0.0 {
                    row /= sum;
                }
            }
        }
        let bmm_pipline = compile_bmm_pipeline(&context);
        let mut encoder_scores =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Use the new batched matmul for Q @ K^T
        run_gpu_bmm(
            &context,
            &mut encoder_scores,
            &bmm_pipline,
            &q_permuted,
            &k_permuted_t,
            &scores_gpu,
            (batch_size * num_heads) as u32, // B
            seq_len as u32,                  // M
            head_dim as u32,                 // K
            seq_len as u32,                  // N
        );
        let softmax_pipeline = compile_softmax_pipeline(&context);
        run_gpu_softmax(
            &context,
            &mut encoder_scores,
            &softmax_pipeline,
            &scores_gpu,
            (batch_size * num_heads * seq_len) as u32,
            seq_len as u32,
            scale,
        );
        context
            .queue
            .submit(std::iter::once(encoder_scores.finish()));
        let gpu_scores_array =
            read_buffer_to_ndarray(&context, &scores_gpu, (num_heads, seq_len, seq_len)).await?;

        println!("Verifying final Attention Scores against CPU implementation...");
        assert_vecs_are_close(
            scores_cpu.as_slice().unwrap(),
            gpu_scores_array.as_slice().unwrap(),
            1e-4,
        );
        println!("✅ Attention Score calculation is correct!");

        Ok(())
    }

    #[tokio::test]
    async fn test_attention_block_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    let (batch_size, seq_len, hidden_size, num_heads) = (1, 16, 64, 4);

    // Create random input data
    let input_cpu: Array3<f32> =
        Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let attention_mask_cpu: Array2<f32> = Array2::ones((batch_size, seq_len));

    // Create random weights
    let q_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
    let q_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
    let k_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
    let k_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
    let v_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
    let v_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
    let out_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
    let out_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

    println!("\n=== DEBUGGING ATTENTION BLOCK ===");
    println!("Input stats: min={:.6}, max={:.6}, mean={:.6}", 
        input_cpu.iter().cloned().fold(f32::INFINITY, f32::min),
        input_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        input_cpu.mean().unwrap()
    );

    // --- GPU Data Setup ---
    let upload_2d = |tensor: &Array2<f32>| -> Arc<wgpu::Buffer> {
        Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(tensor.as_standard_layout().as_slice().unwrap()),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        )
    };
    let upload_1d = |tensor: &Array1<f32>| -> Arc<wgpu::Buffer> {
        Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(tensor.as_slice().unwrap()),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        )
    };

    let gpu_weights = GpuAttentionWeights {
        q_weight: upload_2d(&q_w_cpu.t().to_owned()),
        q_bias: upload_1d(&q_b_cpu),
        k_weight: upload_2d(&k_w_cpu.t().to_owned()),
        k_bias: upload_1d(&k_b_cpu),
        v_weight: upload_2d(&v_w_cpu.t().to_owned()),
        v_bias: upload_1d(&v_b_cpu),
        output_weight: upload_2d(&out_w_cpu.t().to_owned()),
        output_bias: upload_1d(&out_b_cpu),
        norm_weight: upload_1d(&Array1::zeros(hidden_size)),
        norm_bias: upload_1d(&Array1::zeros(hidden_size)),
    };

    let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Attention Input"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Attention Output"),
        size: (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- 2. Act ---

    // == Ground Truth (CPU Path) ==
    println!("\n--- CPU Computation ---");
    let cpu_attention = MultiHeadAttention::new(
        hidden_size,
        num_heads,
        q_w_cpu.t().to_owned(),
        q_b_cpu.clone(),
        k_w_cpu.t().to_owned(),
        k_b_cpu.clone(),
        v_w_cpu.t().to_owned(),
        v_b_cpu.clone(),
        out_w_cpu.t().to_owned(),
        out_b_cpu.clone(),
    );
    let cpu_result = cpu_attention.forward(&input_cpu, None, Some(&attention_mask_cpu))?;
    
    println!("CPU output stats: min={:.6}, max={:.6}, mean={:.6}", 
        cpu_result.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_result.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        cpu_result.mean().unwrap()
    );
    println!("CPU first 10 values: {:?}", &cpu_result.as_slice().unwrap()[..10]);

    // == GPU Path ==
    println!("\n--- GPU Computation ---");
    let mask_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Attention Mask"),
        contents: bytemuck::cast_slice(attention_mask_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Attention Test Encoder"),
    });
    let usage = wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST;
    let mut pipeline: HashMap<Pipeline, Arc<ComputePipeline>> = HashMap::new();
    
    pipeline.insert(Pipeline::MatMul, Arc::new(compile_matmul_pipeline(&context)));
    pipeline.insert(Pipeline::AddBias, Arc::new(compile_add_bias_pipeline(&context)));
    pipeline.insert(Pipeline::Reshape, Arc::new(compile_reshape_pipeline(&context)));
    pipeline.insert(Pipeline::BMM, Arc::new(compile_bmm_pipeline(&context)));
    pipeline.insert(Pipeline::Softmax, Arc::new(compile_softmax_pipeline(&context)));
    pipeline.insert(Pipeline::ApplyMask, Arc::new(compile_apply_mask_pipeline(&context)));
    pipeline.insert(Pipeline::Unreshape, Arc::new(compile_unreshape_pipeline(&context)));
    pipeline.insert(Pipeline::Add, Arc::new(compile_add_pipeline(&context)));
    
    let temp_buffers = {
    let qkv_buffer_size =
        (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64;
    let scores_buffer_size =
        (batch_size * num_heads * seq_len * seq_len * std::mem::size_of::<f32>()) as u64;
    let ffn_intermediate_size = 
        (batch_size * seq_len * hidden_size * 4 * std::mem::size_of::<f32>()) as u64;
    
    AttentionTempBuffers {
        q_proj: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Q Proj"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        }),
        k_proj: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test K Proj"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        }),
        v_proj: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test V Proj"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        }),
        proj_biased: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Proj Biased"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        }),
        q_permuted: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Q Permuted"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        }),
        k_permuted_t: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test K Permuted T"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        }),
        v_permuted: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test V Permuted"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        }),
        scores: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Scores"),
            size: scores_buffer_size,
            usage,
            mapped_at_creation: false,
        }),
        context_vectors: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Context Vectors"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        }),
        ffn_intermediate: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test FFN Intermediate"),
            size: ffn_intermediate_size,
            usage,
            mapped_at_creation: false,
        }),
    }
};
    
    run_gpu_attention_block(
        &context,
        &mut encoder,
        &pipeline,
        &input_gpu,
        &output_gpu,
        &gpu_weights,
        &mask_gpu,
        &temp_buffers,
        batch_size,
        seq_len,
        hidden_size,
        num_heads,
    );
    context.queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely());

    let gpu_result_array =
        read_buffer_to_ndarray(&context, &output_gpu, (batch_size, seq_len, hidden_size))
            .await?;

    println!("GPU output stats: min={:.6}, max={:.6}, mean={:.6}", 
        gpu_result_array.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_result_array.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        gpu_result_array.mean().unwrap()
    );
    println!("GPU first 10 values: {:?}", &gpu_result_array.as_slice().unwrap()[..10]);

    // --- 3. Assert with diagnostics ---
    println!("\n--- Comparison ---");
    let cpu_slice = cpu_result.as_slice().unwrap();
    let gpu_slice = gpu_result_array.as_slice().unwrap();
    
    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut num_nans = 0;
    let mut num_large_diffs = 0;
    
    for (i, (&cpu_val, &gpu_val)) in cpu_slice.iter().zip(gpu_slice.iter()).enumerate() {
        if cpu_val.is_nan() || gpu_val.is_nan() {
            num_nans += 1;
            println!("NaN at index {}: CPU={}, GPU={}", i, cpu_val, gpu_val);
        }
        
        let diff = (cpu_val - gpu_val).abs();
        let rel_diff = if cpu_val.abs() > 1e-6 {
            diff / cpu_val.abs()
        } else {
            diff
        };
        
        max_diff = max_diff.max(diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        
        if diff > 1e-3 {
            num_large_diffs += 1;
            if num_large_diffs <= 5 {  // Print first 5 large differences
                println!("Large diff at index {}: CPU={:.6}, GPU={:.6}, diff={:.6}, rel={:.6}", 
                    i, cpu_val, gpu_val, diff, rel_diff);
            }
        }
    }
    
    println!("\nMax absolute difference: {:.6}", max_diff);
    println!("Max relative difference: {:.6}", max_rel_diff);
    println!("Number of NaNs: {}", num_nans);
    println!("Number of diffs > 1e-3: {} / {}", num_large_diffs, cpu_slice.len());
    
    if num_nans > 0 {
        panic!("Found NaN values in output!");
    }
    
    // Use adaptive tolerance based on magnitude
    let tolerance = if max_diff < 0.01 { 1e-3 } else { 0.01 };
    println!("Using tolerance: {}", tolerance);
    
    assert_vecs_are_close(cpu_slice, gpu_slice, tolerance);
    println!("✅ Attention Block GPU implementation is correct!");

    Ok(())
}
    
}
