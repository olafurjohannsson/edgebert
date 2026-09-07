//! # Decoder Traits

use std::any::Any;

use crate::ChatTemplate;
use crate::cache::Cache;
use crate::common::GenerationConfig;
#[cfg(not(target_arch = "wasm32"))]
use crate::encoder_decoder::traits::GpuCrossAttentionKVCache;
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::{GpuFrameContext, GpuKVCache, GpuTensor, GpuTensorPool};
#[cfg(not(target_arch = "wasm32"))]
use crate::models::base::ModelInput;
use crate::models::base::{AutoregressiveLoop, LanguageModel};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
#[cfg(not(target_arch = "wasm32"))]
use wgpu::CommandEncoder;

/// Defines the generation loop.
#[async_trait]
pub trait DecoderGenerationBackend: Send + Sync {
    /// Token tensor for decode loop (GPU: GpuTensor, CPU: Array2<u32>)
    type DecodeToken: Send + Sync;

    /// Allocate reusable single-token tensor
    fn new_decode_token(&self) -> Result<Self::DecodeToken>;

    /// Update decode tensor with new token ID (fast write)
    fn update_decode_token(&self, tensor: &mut Self::DecodeToken, token_id: u32) -> Result<()>;

    /// Process prompt, populate cache, return first logits
    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        tokens: &Array2<u32>, // Always CPU
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>>;

    /// Prefill continuing from a cache that already holds `start_pos` tokens.
    ///
    /// The default re-prefills everything, so a backend that has not implemented
    /// prefix reuse stays correct and merely gains nothing. Only pass a non-zero
    /// `start_pos` for a cache whose contents are known to match those tokens.
    async fn prefill_at(
        &self,
        model: &dyn DecoderLanguageModel,
        tokens: &Array2<u32>,
        _start_pos: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        self.prefill(model, tokens, cache).await
    }

    /// Process single token, return next logits
    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        token: &Self::DecodeToken,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>>;
}

/// Placeholder GPU decoder trait for wasm builds.
///
/// There is no GPU backend on wasm, but the decoder pipeline still carries
/// `Option<Box<dyn GpuDecoder>>` in its fields and signatures. An empty trait keeps
/// those compiling against a value that can only ever be `None`, which is far less
/// invasive than threading `#[cfg]` through the pipeline. Mirrors what
/// `cpu::encoder::traits::GpuEncoder` does for the encoder path.
#[cfg(target_arch = "wasm32")]
pub trait GpuDecoder: Send + Sync {}

#[cfg(not(target_arch = "wasm32"))]
pub trait GpuDecoder: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;

    fn embed_norm(
        &self,
        _cmd_encoder: &mut CommandEncoder,
        _pool: &mut GpuTensorPool,
        _hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        unimplemented!()
    }
    fn final_norm(
        &self,
        _cmd_encoder: &mut CommandEncoder,
        _pool: &mut GpuTensorPool,
        _hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        unimplemented!()
    }

    fn embed(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor>;

    fn embed_and_normalize(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor>;

    fn forward_layers(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor>;

    fn forward_layers2(
        &self,
        _cmd_encoder: &mut CommandEncoder,
        _pool: &mut GpuTensorPool,
        _hidden_states: &GpuTensor,
        _encoder_hidden_states: &GpuTensor,
        _self_attention_mask: Option<&GpuTensor>,
        _cross_attention_mask: Option<&GpuTensor>,
        _cache: Option<&mut GpuKVCache>,
        _cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
        _start_layer: usize,
        _end_layer: usize,
    ) -> Result<GpuTensor> {
        unimplemented!()
    }

    fn precompute_cross_attention_kv(
        &self,
        _cmd_encoder: &mut CommandEncoder,
        _pool: &mut GpuTensorPool,
        _encoder_hidden_states: &GpuTensor,
    ) -> Result<GpuCrossAttentionKVCache> {
        unimplemented!()
    }

    fn num_layers(&self) -> usize;

    fn hidden_size(&self) -> usize;

    /// Default full forward pass.
    fn forward(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        _encoder_hidden_states: Option<&GpuTensor>, // Reserved for future Seq2Seq reuse
    ) -> Result<GpuTensor> {
        let hidden = self.embed_and_normalize(encoder, pool, input, position_offset)?;

        self.forward_layers(
            encoder,
            pool,
            &hidden,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )
    }
}

pub trait CpuDecoder: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Forward pass through transformer layers
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>>;

    /// Apply final layer normalization.
    fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;

    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_attention_heads(&self) -> usize;

    fn num_kv_heads(&self) -> usize {
        self.num_attention_heads()
    }

    fn head_dim(&self) -> usize {
        self.hidden_size() / self.num_attention_heads()
    }

    fn forward_all_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        self.forward_layers(
            hidden_states,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )
    }

    /// Full decoder pass: layers + final norm.
    fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let output =
            self.forward_all_layers(hidden_states, attention_mask, position_offset, cache)?;
        self.final_norm(&output)
    }
}

pub trait CpuDecoderOps: Send + Sync {
    /// Access the underlying CPU compute component.
    fn decoder(&self) -> &dyn CpuDecoder;

    /// Embed tokens to hidden states.
    fn embed(&self, tokens: &Array2<u32>, pos: usize) -> Result<Array3<f32>>;

    /// Handles model-specific projection logic (e.g., MatMul vs Norm+MatMul).
    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;

    /// Generates the attention mask on the CPU
    fn get_attention_mask(&self, seq_len: usize, past_len: usize) -> Result<Array2<f32>>;

    /// Full forward pass
    fn forward(
        &self,
        tokens: &Array2<u32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let hidden = self.embed(tokens, position_offset)?;
        self.decoder()
            .forward(&hidden, attention_mask, position_offset, cache)
    }

    /// Full inference path: tokens -> hidden -> logits.
    fn forward_to_logits(
        &self,
        tokens: &Array2<u32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let hidden = self.forward(tokens, attention_mask, position_offset, cache)?;
        self.project_to_logits(&hidden)
    }
}

/// Placeholder GPU decoder trait for wasm builds.
///
/// There is no GPU backend on wasm, but the decoder pipeline still carries
/// `Option<Box<dyn GpuDecoderOps>>` in its fields and signatures. An empty trait keeps
/// those compiling against a value that can only ever be `None`, which is far less
/// invasive than threading `#[cfg]` through the pipeline. Mirrors what
/// `cpu::encoder::traits::GpuEncoder` does for the encoder path.
#[cfg(target_arch = "wasm32")]
pub trait GpuDecoderOps: Send + Sync {}

#[cfg(not(target_arch = "wasm32"))]
pub trait GpuDecoderOps: Send + Sync {
    /// Access the underlying GPU compute component.
    fn decoder(&self) -> &dyn GpuDecoder;

    /// Generates the attention mask on the GPU.
    fn get_attention_mask(
        &self,
        ctx: &mut GpuFrameContext,
        seq_len: usize,
        max_len: usize,
    ) -> Result<GpuTensor>;

    /// Handles model-specific projection logic.
    fn project_to_logits(
        &self,
        ctx: &mut GpuFrameContext,
        last_hidden_state: &GpuTensor,
    ) -> Result<GpuTensor>;
}

/// The primary trait for Decoder-only Language Models Llama, GPT, Mistral,
#[async_trait]
pub trait DecoderLanguageModel: LanguageModel {
    /// Access CPU operations strategy. Returns `None` if model is GPU-only.
    fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps>;

    /// Access GPU operations strategy. Returns `None` if model is CPU-only.
    #[cfg(not(target_arch = "wasm32"))]
    fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps>;

    /// Specifies the generation loop strategy (e.g., Pipelined vs Legacy).
    fn autoregressive_loop(&self) -> AutoregressiveLoop;

    /// Returns the default generation config for this specific model type.
    fn get_default_generation_config(&self) -> GenerationConfig {
        GenerationConfig::default()
    }

    /// Get the chat template for this model (if it's an instruct model)
    fn chat_template(&self) -> Option<&dyn ChatTemplate> {
        None // Default: no template (base model)
    }

    /// Check if this model requires a chat template for proper use
    fn is_instruct_model(&self) -> bool {
        self.chat_template().is_some()
    }

    fn get_logits_cpu(&self, text: &str) -> Result<Array3<f32>> {
        let input_ids = self.tokenize(text)?;
        let seq_len = input_ids.ncols();

        let ops = self
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("CPU Ops not available"))?;

        let attention_mask = ops.get_attention_mask(seq_len, 0)?;

        let tokens = ops.embed(&input_ids, 0)?;

        let decoder_output = ops.decoder().forward(&tokens, &attention_mask, 0, None)?;

        ops.project_to_logits(&decoder_output)
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn get_logits_gpu(&self, text: &str) -> Result<Array3<f32>> {
        let input_ids = self.tokenize(text)?;
        // let input_slice = input_ids.as_slice().unwrap();
        let seq_len = input_ids.len();

        let ops = self
            .decoder_gpu_ops()
            .ok_or_else(|| anyhow!("GPU Ops not available"))?;

        let context = self
            .context()
            .ok_or_else(|| anyhow!("Model missing WgpuContext"))?;

        let pool = context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&context, pool_guard);
        let attention_mask = ops.get_attention_mask(&mut frame, seq_len, seq_len)?;
        let (encoder, pool) = frame.resources();
        let hidden = ops.decoder().forward(
            encoder,
            pool,
            ModelInput::TokensCpu(input_ids.view()),
            &attention_mask,
            0,    // offset
            None, // no cache
            None, // no encoder hidden
        )?;

        let logits = ops.project_to_logits(&mut frame, &hidden)?;
        let logits_cpu = logits.to_ndarray_3d::<f32>().await?;

        frame.finish();

        Ok(logits_cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::InferenceModel;
    use std::collections::HashSet;
    use std::sync::Arc;
    use tokenizers::Tokenizer;

    #[derive(Clone)]
    struct MockCache {
        len: usize,
    }

    impl Cache for MockCache {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn get_seq_length(&self) -> usize {
            self.len
        }
        fn set_seq_length(&mut self, len: usize) {
            self.len = len;
        }
        fn clear(&mut self) {
            self.len = 0;
        }
        fn increment_len(&mut self, n: usize) {
            self.len += n;
        }
        fn clone_box(&self) -> Box<dyn Cache> {
            Box::new(self.clone())
        }
    }

    struct MockCpuDecoder {
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
    }

    impl MockCpuDecoder {
        fn new(num_layers: usize, hidden_size: usize, num_heads: usize) -> Self {
            Self {
                num_layers,
                hidden_size,
                num_heads,
            }
        }
    }

    impl CpuDecoder for MockCpuDecoder {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn forward_layers(
            &self,
            hidden_states: &Array3<f32>,
            _attention_mask: &Array2<f32>,
            _position_offset: usize,
            _cache: Option<&mut dyn Cache>,
            _start_layer: usize,
            _end_layer: usize,
        ) -> Result<Array3<f32>> {
            Ok(hidden_states.clone())
        }

        fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            Ok(hidden_states.clone())
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }
        fn hidden_size(&self) -> usize {
            self.hidden_size
        }
        fn num_attention_heads(&self) -> usize {
            self.num_heads
        }
    }

    #[test]
    fn test_cpu_decoder_metadata() {
        let decoder = MockCpuDecoder::new(12, 768, 12);

        assert_eq!(decoder.num_layers(), 12);
        assert_eq!(decoder.hidden_size(), 768);
        assert_eq!(decoder.num_attention_heads(), 12);
    }

    #[test]
    fn test_cpu_decoder_default_num_kv_heads() {
        let decoder = MockCpuDecoder::new(12, 768, 12);

        assert_eq!(decoder.num_kv_heads(), 12);
    }

    #[test]
    fn test_cpu_decoder_default_head_dim() {
        let decoder = MockCpuDecoder::new(12, 768, 12);

        assert_eq!(decoder.head_dim(), 64); // 768 / 12
    }

    #[test]
    fn test_cpu_decoder_head_dim_different_configs() {
        // Llama-style: 4096 hidden, 32 heads
        let decoder1 = MockCpuDecoder::new(32, 4096, 32);
        assert_eq!(decoder1.head_dim(), 128); // 4096 / 32

        // GPT-2 small: 768 hidden, 12 heads
        let decoder2 = MockCpuDecoder::new(12, 768, 12);
        assert_eq!(decoder2.head_dim(), 64); // 768 / 12

        // Custom: 512 hidden, 8 heads
        let decoder3 = MockCpuDecoder::new(6, 512, 8);
        assert_eq!(decoder3.head_dim(), 64); // 512 / 8
    }

    #[test]
    fn test_cpu_decoder_forward_layers() {
        let decoder = MockCpuDecoder::new(6, 64, 4);

        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));

        let output = decoder
            .forward_layers(&hidden, &mask, 0, None, 0, 6)
            .unwrap();

        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_forward_all_layers() {
        let decoder = MockCpuDecoder::new(6, 64, 4);

        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));

        let output = decoder.forward_all_layers(&hidden, &mask, 0, None).unwrap();

        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_forward_default() {
        let decoder = MockCpuDecoder::new(6, 64, 4);

        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));

        let output = decoder.forward(&hidden, &mask, 0, None).unwrap();

        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_final_norm() {
        let decoder = MockCpuDecoder::new(6, 64, 4);

        let hidden = Array3::<f32>::ones((1, 5, 64));

        let output = decoder.final_norm(&hidden).unwrap();

        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_with_cache() {
        let decoder = MockCpuDecoder::new(6, 64, 4);

        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));
        let mut cache = MockCache { len: 0 };

        let output = decoder
            .forward(&hidden, &mask, 0, Some(&mut cache))
            .unwrap();

        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_partial_layers() {
        let decoder = MockCpuDecoder::new(12, 64, 4);

        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));

        // Run only first half
        let output1 = decoder
            .forward_layers(&hidden, &mask, 0, None, 0, 6)
            .unwrap();
        assert_eq!(output1.shape(), &[1, 5, 64]);

        // Run only second half
        let output2 = decoder
            .forward_layers(&hidden, &mask, 0, None, 6, 12)
            .unwrap();
        assert_eq!(output2.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_position_offset() {
        let decoder = MockCpuDecoder::new(6, 64, 4);

        let hidden = Array3::<f32>::ones((1, 1, 64)); // Single token
        let mask = Array2::<f32>::ones((1, 10)); // Mask for position 10

        // Simulate decoding at position 9 (0-indexed)
        let output = decoder.forward(&hidden, &mask, 9, None).unwrap();

        assert_eq!(output.shape(), &[1, 1, 64]);
    }

    #[test]
    fn test_cpu_decoder_as_any() {
        let decoder = MockCpuDecoder::new(6, 64, 4);

        let any_ref = decoder.as_any();
        let downcasted = any_ref.downcast_ref::<MockCpuDecoder>();

        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().num_layers, 6);
    }

    #[test]
    fn test_cpu_decoder_as_any_mut() {
        let mut decoder = MockCpuDecoder::new(6, 64, 4);

        {
            let any_mut = decoder.as_any_mut();
            let downcasted = any_mut.downcast_mut::<MockCpuDecoder>();
            assert!(downcasted.is_some());
            downcasted.unwrap().num_layers = 12;
        }

        assert_eq!(decoder.num_layers, 12);
    }

    struct MockCpuDecoderOps {
        decoder: MockCpuDecoder,
    }

    impl MockCpuDecoderOps {
        fn new() -> Self {
            Self {
                decoder: MockCpuDecoder::new(6, 64, 4),
            }
        }
    }

    impl CpuDecoderOps for MockCpuDecoderOps {
        fn decoder(&self) -> &dyn CpuDecoder {
            &self.decoder
        }

        fn embed(&self, tokens: &Array2<u32>, _pos: usize) -> Result<Array3<f32>> {
            let (batch, seq_len) = tokens.dim();
            Ok(Array3::<f32>::ones((batch, seq_len, 64)))
        }

        fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            let (batch, seq_len, _) = hidden_states.dim();
            Ok(Array3::<f32>::zeros((batch, seq_len, 1000))) // vocab_size = 1000
        }

        fn get_attention_mask(&self, seq_len: usize, _past_len: usize) -> Result<Array2<f32>> {
            Ok(Array2::<f32>::ones((seq_len, seq_len)))
        }
    }

    #[test]
    fn test_cpu_decoder_ops_decoder_access() {
        let ops = MockCpuDecoderOps::new();

        let decoder = ops.decoder();
        assert_eq!(decoder.num_layers(), 6);
        assert_eq!(decoder.hidden_size(), 64);
    }

    #[test]
    fn test_cpu_decoder_ops_embed() {
        let ops = MockCpuDecoderOps::new();

        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();
        let hidden = ops.embed(&tokens, 0).unwrap();

        assert_eq!(hidden.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_ops_project_to_logits() {
        let ops = MockCpuDecoderOps::new();

        let hidden = Array3::<f32>::ones((1, 5, 64));
        let logits = ops.project_to_logits(&hidden).unwrap();

        assert_eq!(logits.shape(), &[1, 5, 1000]);
    }

    #[test]
    fn test_cpu_decoder_ops_get_attention_mask() {
        let ops = MockCpuDecoderOps::new();

        let mask = ops.get_attention_mask(10, 0).unwrap();

        assert_eq!(mask.shape(), &[10, 10]);
    }

    #[test]
    fn test_cpu_decoder_ops_forward_default() {
        let ops = MockCpuDecoderOps::new();

        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();
        let mask = Array2::<f32>::ones((5, 5));

        // Uses default implementation
        let hidden = ops.forward(&tokens, &mask, 0, None).unwrap();

        assert_eq!(hidden.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_ops_forward_to_logits_default() {
        let ops = MockCpuDecoderOps::new();

        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();
        let mask = Array2::<f32>::ones((5, 5));

        // Uses default implementation: forward + project_to_logits
        let logits = ops.forward_to_logits(&tokens, &mask, 0, None).unwrap();

        assert_eq!(logits.shape(), &[1, 5, 1000]);
    }

    struct MockDecoderModel {
        cpu_ops: MockCpuDecoderOps,
    }

    impl InferenceModel for MockDecoderModel {
        fn device(&self) -> crate::traits::Device {
            crate::traits::Device::Cpu
        }
        fn context(&self) -> Option<Arc<crate::WgpuContext>> {
            None
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl LanguageModel for MockDecoderModel {
        fn vocab_size(&self) -> usize {
            1000
        }
        fn hidden_size(&self) -> usize {
            64
        }
        fn num_layers(&self) -> usize {
            6
        }
        fn num_heads(&self) -> usize {
            4
        }
        fn context_size(&self) -> usize {
            2048
        }
        fn tokenizer(&self) -> &Tokenizer {
            unimplemented!()
        }
        fn eos_token_id(&self) -> Option<u32> {
            Some(2)
        }
        fn bos_token_id(&self) -> Option<u32> {
            Some(1)
        }
        fn forced_bos_token_id(&self) -> Option<u32> {
            None
        }
        fn forced_eos_token_id(&self) -> Option<u32> {
            None
        }
        fn pad_token_id(&self) -> Option<u32> {
            Some(0)
        }
        fn stop_token_ids(&self) -> HashSet<u32> {
            HashSet::from([2])
        }
        fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
            Ok(Box::new(MockCache { len: 0 }))
        }
    }

    #[async_trait]
    impl DecoderLanguageModel for MockDecoderModel {
        fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
            Some(&self.cpu_ops)
        }

        fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
            None
        }

        fn autoregressive_loop(&self) -> AutoregressiveLoop {
            AutoregressiveLoop::Pipelined
        }
    }

    #[test]
    fn test_decoder_language_model_get_default_generation_config() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };

        let config = model.get_default_generation_config();

        // Should return default config
        assert!(config.max_length > 0);
    }

    #[test]
    fn test_decoder_language_model_chat_template_default() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };

        // Default: no chat template
        assert!(model.chat_template().is_none());
    }

    #[test]
    fn test_decoder_language_model_is_instruct_model_default() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };

        assert!(!model.is_instruct_model());
    }

    #[test]
    fn test_decoder_language_model_cpu_ops_access() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };

        let ops = model.decoder_cpu_ops();
        assert!(ops.is_some());

        let ops = ops.unwrap();
        assert_eq!(ops.decoder().num_layers(), 6);
    }

    #[test]
    fn test_decoder_language_model_gpu_ops_none() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };

        // CPU-only model
        assert!(model.decoder_gpu_ops().is_none());
    }

    #[test]
    fn test_decoder_language_model_autoregressive_loop() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };

        assert!(matches!(
            model.autoregressive_loop(),
            AutoregressiveLoop::Pipelined
        ));
    }

    #[test]
    fn test_autoregressive_loop_variants() {
        let pipelined = AutoregressiveLoop::Pipelined;
        let legacy = AutoregressiveLoop::Legacy;

        // They should be different
        assert!(!matches!(pipelined, AutoregressiveLoop::Legacy));
        assert!(!matches!(legacy, AutoregressiveLoop::Pipelined));
    }
    #[test]
    fn test_cpu_decoder_empty_sequence() {
        let decoder = MockCpuDecoder::new(6, 64, 4);

        let hidden = Array3::<f32>::zeros((1, 0, 64)); // Empty sequence
        let mask = Array2::<f32>::zeros((0, 0));

        let output = decoder.forward(&hidden, &mask, 0, None).unwrap();

        assert_eq!(output.shape(), &[1, 0, 64]);
    }

    #[test]
    fn test_cpu_decoder_batch_size_greater_than_one() {
        let decoder = MockCpuDecoder::new(6, 64, 4);

        let hidden = Array3::<f32>::ones((4, 5, 64)); // batch = 4
        let mask = Array2::<f32>::ones((5, 5));

        let output = decoder.forward(&hidden, &mask, 0, None).unwrap();

        assert_eq!(output.shape(), &[4, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_ops_embed_batch() {
        let ops = MockCpuDecoderOps::new();

        let tokens = Array2::from_shape_vec((2, 3), vec![1u32, 2, 3, 4, 5, 6]).unwrap();
        let hidden = ops.embed(&tokens, 0).unwrap();

        assert_eq!(hidden.shape(), &[2, 3, 64]);
    }

    #[test]
    fn test_cpu_decoder_single_layer() {
        let decoder = MockCpuDecoder::new(1, 64, 4);

        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));

        let output = decoder.forward(&hidden, &mask, 0, None).unwrap();

        assert_eq!(output.shape(), &[1, 5, 64]);
        assert_eq!(decoder.num_layers(), 1);
    }

    #[test]
    fn test_cpu_decoder_large_hidden_size() {
        let decoder = MockCpuDecoder::new(32, 4096, 32);

        assert_eq!(decoder.hidden_size(), 4096);
        assert_eq!(decoder.head_dim(), 128);
    }

    #[test]
    fn test_attention_mask_generation_prefill() {
        let ops = MockCpuDecoderOps::new();

        // Prefill: seq_len = 10, past_len = 0
        let mask = ops.get_attention_mask(10, 0).unwrap();

        assert_eq!(mask.shape(), &[10, 10]);
    }

    #[test]
    fn test_attention_mask_generation_decode() {
        let ops = MockCpuDecoderOps::new();

        // Decode: seq_len = 1, past_len = 10
        let mask = ops.get_attention_mask(1, 10).unwrap();

        assert_eq!(mask.shape(), &[1, 1]);
    }

    #[test]
    fn test_full_cpu_inference_pipeline() {
        let ops = MockCpuDecoderOps::new();

        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 100, 200, 300, 2]).unwrap();

        let mask = ops.get_attention_mask(5, 0).unwrap();

        let hidden = ops.embed(&tokens, 0).unwrap();
        assert_eq!(hidden.shape(), &[1, 5, 64]);

        let output = ops.decoder().forward(&hidden, &mask, 0, None).unwrap();
        assert_eq!(output.shape(), &[1, 5, 64]);

        let logits = ops.project_to_logits(&output).unwrap();
        assert_eq!(logits.shape(), &[1, 5, 1000]);
    }

    #[test]
    fn test_full_cpu_inference_with_forward_to_logits() {
        let ops = MockCpuDecoderOps::new();

        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 100, 200, 300, 2]).unwrap();
        let mask = ops.get_attention_mask(5, 0).unwrap();

        let logits = ops.forward_to_logits(&tokens, &mask, 0, None).unwrap();

        assert_eq!(logits.shape(), &[1, 5, 1000]);
    }
}
