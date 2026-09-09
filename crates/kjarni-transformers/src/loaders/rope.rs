#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::{WgpuContext, gpu_ops::blocks::rope::GpuRoPE};
use crate::{rope::RoPE, traits::ModelMetadata};
use anyhow::Result;
use std::sync::Arc;

pub struct LoadedRoPE {
    pub cpu: Arc<RoPE>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub gpu: Option<Arc<GpuRoPE>>,
}

impl LoadedRoPE {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn new(
        ctx: Option<&Arc<WgpuContext>>,
        meta: &ModelMetadata,
        load_gpu: bool,
    ) -> Result<Self> {
        let cpu_rope = Arc::new(RoPE::new_with_scaling(
            meta.head_dim,
            meta.max_seq_len,
            meta.rope_theta.unwrap_or(10000.0),
            meta.rope_scaling.as_ref(),
        ));

        let gpu = if load_gpu {
            let c = ctx.ok_or_else(|| anyhow::anyhow!("GPU RoPE requires Context"))?;
            Some(Arc::new(GpuRoPE::new(
                c,
                &cpu_rope.cos_cache,
                &cpu_rope.sin_cache,
            )?))
        } else {
            None
        };

        Ok(Self { cpu: cpu_rope, gpu })
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
    pub fn new(meta: &ModelMetadata) -> Result<Self> {
        let cpu_rope = Arc::new(RoPE::new_with_scaling(
            meta.head_dim,
            meta.max_seq_len,
            meta.rope_theta.unwrap_or(10000.0),
            meta.rope_scaling.as_ref(),
        ));

        Ok(Self { cpu: cpu_rope })
    }
}

#[cfg(test)]
mod loaded_rope {

    use super::*;
    use crate::WgpuContext;
    use crate::gpu::GpuTensor;
    use crate::traits::ModelMetadata;
    use anyhow::Result;
    use ndarray::Array4;
    use std::sync::Arc;

    fn dummy_metadata() -> ModelMetadata {
        ModelMetadata {
            hidden_size: 64,
            num_attention_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            max_seq_len: 128,
            rope_theta: Some(10000.0),
            rope_scaling: None,
            decoder_layers: None,
            intermediate_size: 0,
            num_layers: 1,
            vocab_size: 100,
            norm_eps: 1e-5,
            activation: crate::activations::Activation::Gelu,
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: true,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            problem_type: None,
            normalization_strategy: crate::traits::NormalizationStrategy::RMSNorm,
            no_scale_qk: false,
        }
    }

    fn assert_tensors_close(a: &Array4<f32>, b: &Array4<f32>, eps: f32) {
        assert_eq!(a.shape(), b.shape());
        for (v1, v2) in a.iter().zip(b.iter()) {
            assert!((v1 - v2).abs() < eps, "Mismatch: {} vs {}", v1, v2);
        }
    }

    #[test]
    fn test_loaded_rope_cpu_only() {
        let meta = dummy_metadata();

        let loaded = LoadedRoPE::new(None, &meta, false).unwrap();

        assert!(loaded.gpu.is_none());
        assert_eq!(loaded.cpu.head_dim, 32);
        assert_eq!(loaded.cpu.cos_cache.shape(), &[128, 32]);
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_loaded_rope_gpu_init() -> Result<()> {
        let ctx = Arc::new(WgpuContext::new().await?);
        let meta = dummy_metadata();

        let loaded = LoadedRoPE::new(Some(&ctx), &meta, true)?;

        assert!(loaded.gpu.is_some());
        // CPU should always be loaded as source of truth
        assert_eq!(loaded.cpu.head_dim, 32);
        Ok(())
    }

    #[test]
    fn test_loaded_rope_scaling_propagation() {
        let mut meta = dummy_metadata();
        // Use dummy llama3 scaling
        meta.rope_scaling = Some(crate::models::base::RopeScalingConfig {
            rope_type: "llama3".to_string(),
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
            long_factor: None,
            short_factor: None,
        });

        let loaded = LoadedRoPE::new(None, &meta, false).unwrap();

        assert!((loaded.cpu.cos_cache[[0, 0]] - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_rope_gpu_cpu_parity() -> Result<()> {
        let ctx = Arc::new(WgpuContext::new().await?);
        let meta = dummy_metadata();

        let loaded = LoadedRoPE::new(Some(&ctx), &meta, true)?;
        let gpu_rope = loaded.gpu.as_ref().unwrap();

        let shape = [1, 2, 4, 32];
        let size = 2 * 4 * 32;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let input_cpu = Array4::from_shape_vec(shape, data)?;
        let input_gpu = GpuTensor::from_ndarray(&ctx, &input_cpu)?;
        let output_gpu = GpuTensor::zeros(
            &ctx,
            input_cpu.shape().to_vec(),
            crate::tensor::DType::F32,
            "zeros",
        )?;

        let cpu_rotated = loaded.cpu.rotate_4d(&input_cpu, 0);

        let mut encoder = ctx.device.create_command_encoder(&Default::default());

        gpu_rope.encode(
            &mut encoder,
            &input_gpu,
            &output_gpu,
            0, // offset
        );

        ctx.queue.submit(Some(encoder.finish()));

        let gpu_result = output_gpu
            .to_ndarray_4d::<f32>()
            .await?
            .into_dimensionality::<ndarray::Ix4>()?;

        // 4. Compare
        assert_tensors_close(&cpu_rotated, &gpu_result, 1e-5);

        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_rope_gpu_offset_parity() -> Result<()> {
        let ctx = WgpuContext::new().await?;
        let meta = dummy_metadata();
        let loaded = LoadedRoPE::new(Some(&ctx), &meta, true)?;

        let shape = [1, 1, 1, 32];
        let input_cpu = Array4::from_elem(shape, 1.0);
        let input_gpu = GpuTensor::from_ndarray(&ctx, &input_cpu)?;
        let output_gpu = GpuTensor::zeros(
            &ctx,
            input_cpu.shape().to_vec(),
            crate::tensor::DType::F32,
            "zeros",
        )?;

        let offset = 10;
        let cpu_rotated = loaded.cpu.rotate_4d(&input_cpu, offset);
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        loaded
            .gpu
            .as_ref()
            .unwrap()
            .encode(&mut encoder, &input_gpu, &output_gpu, offset);
        ctx.queue.submit(Some(encoder.finish()));

        let gpu_result = output_gpu
            .to_ndarray_4d::<f32>()
            .await?
            .into_dimensionality::<ndarray::Ix4>()?;

        assert_tensors_close(&cpu_rotated, &gpu_result, 1e-5);
        Ok(())
    }
}
