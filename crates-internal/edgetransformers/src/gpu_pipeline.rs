//! A reusable orchestrator for running a generic transformer encoder pipeline on the GPU.

use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, s};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, CommandEncoder, include_wgsl};

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
        attention_mask: &Array2<f32>,
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
        let mask_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Attention Mask Buffer"),
            contents: bytemuck::cast_slice(attention_mask.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
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
                &mask_buffer,
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
    mask: &Buffer,
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
    let temp_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Temp A"),
        size: qkv_buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let temp_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Temp B"),
        size: qkv_buffer_size,
        usage,
        mapped_at_creation: false,
    });
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
    run_gpu_matmul(context, encoder, input, &weights.q_weight, &temp_a, m, k, k);
    run_gpu_add_bias(context, encoder, &temp_a, &weights.q_bias, &temp_b, m * k);
    run_gpu_reshape(
        context,
        encoder,
        &temp_b,
        &q_permuted,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        false,
    );

    // K Path: input -> temp_a -> temp_b -> k_permuted_t
    run_gpu_matmul(context, encoder, input, &weights.k_weight, &temp_a, m, k, k);
    run_gpu_add_bias(context, encoder, &temp_a, &weights.k_bias, &temp_b, m * k);
    run_gpu_reshape(
        context,
        encoder,
        &temp_b,
        &k_permuted_t,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        true,
    );

    // V Path: input -> temp_a -> temp_b -> v_permuted
    run_gpu_matmul(context, encoder, input, &weights.v_weight, &temp_a, m, k, k);
    run_gpu_add_bias(context, encoder, &temp_a, &weights.v_bias, &temp_b, m * k);
    run_gpu_reshape(
        context,
        encoder,
        &temp_b,
        &v_permuted,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        false,
    );

    // --- 3. Attention Scores: Q @ K^T ---
    run_gpu_bmm(
        context,
        encoder,
        &q_permuted,
        &k_permuted_t,
        &scores,
        (batch_size * num_heads) as u32,
        seq_len as u32,
        head_dim as u32,
        seq_len as u32,
    );

    run_gpu_apply_mask(
        context,
        encoder,
        &scores,
        mask,
        batch_size as u32,
        num_heads as u32,
        seq_len as u32,
    );

    // 4. Softmax (in-place on scores)
    let scale = 1.0 / (head_dim as f32).sqrt();
    run_gpu_softmax(
        context,
        encoder,
        &scores,
        (batch_size * num_heads * seq_len) as u32,
        seq_len as u32,
        scale,
    );

    // 5. Apply Scores to V: Scores @ V -> context_vectors
    run_gpu_bmm(
        context,
        encoder,
        &scores,
        &v_permuted,
        &context_vectors,
        (batch_size * num_heads) as u32,
        seq_len as u32,
        seq_len as u32,
        head_dim as u32,
    );

    // 6. "Un-reshape" and Output Projection
    run_gpu_unreshape(
        context,
        encoder,
        &context_vectors,
        &temp_a,
        batch_size as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
    );
    run_gpu_matmul(
        context,
        encoder,
        &temp_a,
        &weights.output_weight,
        &temp_b,
        m,
        k,
        k,
    );
    run_gpu_add_bias(
        context,
        encoder,
        &temp_b,
        &weights.output_bias,
        output,
        m * k,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeedForward, LayerNorm, wgpu_context::WgpuContext};
    use ndarray::{Array, Array1, Array2, Array3};
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
        run_gpu_matmul(
            &context,
            &mut encoder,
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
            &q_proj_gpu,
            &q_bias_gpu,
            &q_biased_gpu,
            (batch_size * seq_len * hidden_size) as u32,
        );
        run_gpu_reshape(
            &context,
            &mut encoder,
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
        run_gpu_matmul(
            &context,
            &mut encoder_q,
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
            &q_proj,
            &q_bias_gpu,
            &proj_biased,
            (seq_len * hidden_size * batch_size) as u32,
        );
        run_gpu_reshape(
            &context,
            &mut encoder_q,
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
            &k_proj,
            &k_bias_gpu,
            &proj_biased,
            (seq_len * hidden_size * batch_size) as u32,
        );
        run_gpu_reshape(
            &context,
            &mut encoder_k,
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

        let mut encoder_scores =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Use the new batched matmul for Q @ K^T
        run_gpu_bmm(
            &context,
            &mut encoder_scores,
            &q_permuted,
            &k_permuted_t,
            &scores_gpu,
            (batch_size * num_heads) as u32, // B
            seq_len as u32,                  // M
            head_dim as u32,                 // K
            seq_len as u32,                  // N
        );

        run_gpu_softmax(
            &context,
            &mut encoder_scores,
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

        // Create random weights, ensuring they are identical for both CPU and GPU paths.
        // Weights are created in the format they would be loaded from a file (e.g., PyTorch's [out, in]).
        let q_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
        let q_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
        let k_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
        let k_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
        let v_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
        let v_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
        let out_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
        let out_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

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

        // The GPU path expects pre-transposed weights, just like the CPU path.
        // This simulates the logic from `GpuTransformerEncoder::new`.
        let gpu_weights = GpuAttentionWeights {
            q_weight: upload_2d(&q_w_cpu.t().to_owned()),
            q_bias: upload_1d(&q_b_cpu),
            k_weight: upload_2d(&k_w_cpu.t().to_owned()),
            k_bias: upload_1d(&k_b_cpu),
            v_weight: upload_2d(&v_w_cpu.t().to_owned()),
            v_bias: upload_1d(&v_b_cpu),
            output_weight: upload_2d(&out_w_cpu.t().to_owned()),
            output_bias: upload_1d(&out_b_cpu),
            // Norm weights are not used by this specific function, so they can be dummy values.
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

        // == GPU Path ==
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Attention Test Encoder"),
        });
        run_gpu_attention_block(
            &context,
            &mut encoder,
            &input_gpu,
            &output_gpu,
            &gpu_weights,
            batch_size,
            seq_len,
            hidden_size,
            num_heads,
        );
        context.queue.submit(std::iter::once(encoder.finish()));

        let gpu_result_array =
            read_buffer_to_ndarray(&context, &output_gpu, (batch_size, seq_len, hidden_size))
                .await?;

        // --- 3. Assert ---
        println!("Verifying Attention Block GPU kernel against CPU implementation...");
        assert_vecs_are_close(
            cpu_result.as_slice().unwrap(),
            gpu_result_array.as_slice().unwrap(),
            1e-3, // Use a slightly higher tolerance for the complex attention calculation
        );
        println!("✅ Attention Block GPU implementation is correct!");

        Ok(())
    }
}
