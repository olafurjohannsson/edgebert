//! Base traits for language models
//!
//! This module provides high-level, user-facing traits that abstract over
//! the low-level architecture traits in `traits.rs`.

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use std::sync::Arc;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::WgpuContext;
use crate::cpu::encoder::buffers::EncoderBuffers;
use crate::cpu::encoder::config::{EncodingConfig, PoolingStrategy};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::gpu::{GpuFrameContext, GpuTensor, GpuTensorPool};
use crate::models::base::LanguageModel;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::models::base::ModelInput;
use crate::pooling::mean_pool;
use crate::traits::CpuTransformerCore;
use crate::{last_token_pool, max_pool};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};

/// Classification mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClassificationMode {
    /// Single-label classification (softmax, mutually exclusive).
    #[default]
    SingleLabel,

    /// Multi-label classification (sigmoid, independent labels).
    MultiLabel,
}

/// Trait for encoder-only language models (BERT, RoBERTa, etc.)
#[async_trait]
pub trait EncoderLanguageModel: LanguageModel {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps>;

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps>;

    fn dimension(&self) -> usize {
        self.hidden_size()
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn encoder_dimensions(&self) -> usize {
        // either GPU or CPU
        match self.device() {
            crate::traits::Device::Cpu => {
                let ops = self
                    .encoder_cpu_ops()
                    .expect("CPU ops not implemented for this model");
                ops.encoder().hidden_size()
            }
            crate::traits::Device::Wgpu => {
                let ops = self
                    .encoder_gpu_ops()
                    .expect("GPU ops not implemented for this model");
                ops.encoder().hidden_size()
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
    fn encoder_dimensions(&self) -> usize {
        let ops = self
            .encoder_cpu_ops()
            .expect("CPU ops not implemented for this model");
        ops.encoder().hidden_size()
    }

    /// Get hidden states for input text
    async fn get_hidden_states(&self, text: &str) -> Result<Array3<f32>> {
        let (batch_hidden_states, _) = self.get_hidden_states_batch(&[text]).await?;
        Ok(batch_hidden_states)
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    async fn get_hidden_states_batch_from_ids(
        &self,
        input_ids: &Array2<u32>,
        attention_mask: &Array2<u32>,
    ) -> Result<(Array3<f32>, Array2<f32>)> {
        let attention_mask_f32 = attention_mask.mapv(|x| x as f32);
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_states = if let Some(ops) = self.encoder_cpu_ops() {
            let encoder: &dyn CpuEncoder = ops.encoder();

            let hidden: Array3<f32> = ops.embed_tokens(input_ids, None, 0)?;
            let normalized_hidden: Array3<f32> = encoder.embed_norm(&hidden)?;

            {
                let mut buffers = encoder.create_buffers(batch_size, seq_len);
                #[cfg(debug_assertions)]
                {
                    let buf_desc: String = buffers.memory_breakdown();
                    println!(
                        "Encoder buffers allocated for batch_size={}, seq_len={}: {}",
                        batch_size, seq_len, buf_desc
                    );
                }
                encoder
                    .forward_with_buffers(&normalized_hidden, &attention_mask_f32, &mut buffers)?
                    .last_hidden_state
            }
        } else if let Some(ops) = self.encoder_gpu_ops() {
            let context = self
                .context()
                .ok_or_else(|| anyhow!("GPU model missing context"))?;

            let pool: std::sync::Arc<tokio::sync::Mutex<GpuTensorPool>> =
                context.get_inference_pool();

            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&context, pool_guard);
            let (encoder_cmd, pool_ref) = frame.resources();

            // Upload data to GPU
            let input_ids_gpu = GpuTensor::from_ndarray(&context, input_ids)?;
            let attention_mask_gpu = GpuTensor::from_ndarray(&context, &attention_mask_f32)?;

            // Run the forward pass
            #[allow(
                deprecated,
                reason = "internal caller of an API deprecated without a named replacement"
            )]
            let gpu_output = ops.encoder().forward(
                encoder_cmd,
                pool_ref,
                ModelInput::TokensGpu(&input_ids_gpu),
                &attention_mask_gpu,
                None, // token_type_ids
            )?;

            frame.finish();

            // Download the result back to CPU
            gpu_output.last_hidden_state.to_ndarray_3d().await?
        } else {
            return Err(anyhow!(
                "No available CPU or GPU encoder implementation for this model."
            ));
        };

        Ok((hidden_states, attention_mask_f32))
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
    async fn get_hidden_states_batch_from_ids(
        &self,
        input_ids: &Array2<u32>,
        attention_mask: &Array2<u32>,
    ) -> Result<(Array3<f32>, Array2<f32>)> {
        let attention_mask_f32 = attention_mask.mapv(|x| x as f32);
        let (batch_size, seq_len) = input_ids.dim();
        let ops = self
            .encoder_cpu_ops()
            .ok_or_else(|| anyhow!("No available CPU encoder implementation for this model."))?;
        let encoder: &dyn CpuEncoder = ops.encoder();

        let hidden: Array3<f32> = ops.embed_tokens(input_ids, None, 0)?;
        let normalized_hidden: Array3<f32> = encoder.embed_norm(&hidden)?;

        let hidden_states = {
            let mut buffers = encoder.create_buffers(batch_size, seq_len);
            encoder
                .forward_with_buffers(&normalized_hidden, &attention_mask_f32, &mut buffers)?
                .last_hidden_state
        };

        Ok((hidden_states, attention_mask_f32))
    }

    fn encode_batch_texts(&self, texts: &[&str]) -> Result<Vec<tokenizers::Encoding>> {
        self.tokenizer()
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("Tokenizer failed: {}", e))
    }
    /// Get hidden states for a batch of texts
    async fn get_hidden_states_batch(&self, texts: &[&str]) -> Result<(Array3<f32>, Array2<f32>)> {
        if texts.is_empty() {
            return Ok((
                Array3::zeros((0, 0, self.hidden_size())),
                Array2::zeros((0, 0)),
            ));
        }

        let encodings = self.encode_batch_texts(texts)?;

        let batch_size = encodings.len();
        if batch_size == 0 {
            return Ok((
                Array3::zeros((0, 0, self.hidden_size())),
                Array2::zeros((0, 0)),
            ));
        }
        let sequence_length = encodings[0].len();

        let input_ids_vec: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_ids())
            .cloned()
            .collect();
        let attention_mask_vec: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask())
            .cloned()
            .collect();

        let input_ids = Array2::from_shape_vec((batch_size, sequence_length), input_ids_vec)?;
        let attention_mask =
            Array2::from_shape_vec((batch_size, sequence_length), attention_mask_vec)?;

        self.get_hidden_states_batch_from_ids(&input_ids, &attention_mask)
            .await
    }
}

/// Defines the application of an encoder model for sentence similarity tasks
#[async_trait]
pub trait SentenceEncoderModel: EncoderLanguageModel {
    /// Encode a batch of texts into embedding vectors.
    async fn encode_batch(&self, texts: &[&str], config: &EncodingConfig) -> Result<Vec<Vec<f32>>>;

    /// Encode a single text into an embedding vector.
    async fn encode(&self, text: &str, config: &EncodingConfig) -> Result<Vec<f32>> {
        let batch_result = self.encode_batch(&[text], config).await?;
        batch_result
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Batch encoding returned no results for a single item."))
    }
}

#[async_trait]
impl<T: EncoderLanguageModel + Sync> SentenceEncoderModel for T {
    async fn encode_batch(&self, texts: &[&str], config: &EncodingConfig) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let (hidden_states, attention_mask) = self.get_hidden_states_batch(texts).await?;

        let mut pooled = match config.pooling_strategy {
            PoolingStrategy::Cls => hidden_states.slice(ndarray::s![.., 0, ..]).to_owned(),
            PoolingStrategy::Mean => mean_pool(&hidden_states, &attention_mask)?,
            PoolingStrategy::LastToken => last_token_pool(&hidden_states, &attention_mask)?,
            PoolingStrategy::Max => max_pool(&hidden_states, &attention_mask)?,
        };

        if config.normalize {
            l2_normalize_inplace(&mut pooled);
        }

        Ok(pooled.outer_iter().map(|row| row.to_vec()).collect())
    }
}
/// Output from a CPU encoder.
#[derive(Clone, Debug)]
pub struct CpuEncoderOutput {
    pub last_hidden_state: Array3<f32>,
}

/// CPU-based transformer encoder trait
/// ```
pub trait CpuEncoder: CpuTransformerCore {
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>>;

    fn create_buffers(&self, max_batch: usize, max_seq: usize) -> EncoderBuffers;

    fn forward_with_buffers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        _buffers: &mut EncoderBuffers,
    ) -> Result<CpuEncoderOutput> {
        // Default: ignore buffers, call regular forward
        let hidden = self.forward_layers(hidden_states, attention_mask, 0, self.num_layers());
        Ok(CpuEncoderOutput {
            last_hidden_state: hidden?,
        })
    }

    /// Full forward: layers -> final_norm
    fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
    ) -> Result<CpuEncoderOutput> {
        let output = self.forward_layers(hidden_states, attention_mask, 0, self.num_layers())?;
        // let normalized = self.final_norm(&output)?;
        Ok(CpuEncoderOutput {
            last_hidden_state: output,
        })
    }
}

pub trait CpuEncoderOps: Send + Sync {
    /// Access the underlying encoder (transformer layers).
    fn encoder(&self) -> &dyn CpuEncoder;

    /// Embed tokens to hidden states, no normalization happens
    fn embed_tokens(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
        pos: usize,
    ) -> Result<Array3<f32>>;

    /// Embed audio -> hidden states, no normalization happens
    fn embed_audio(&self, _mel: &Array3<f32>) -> Result<Array3<f32>> {
        Err(anyhow!("Audio embedding not supported for this model"))
    }

    fn get_attention_mask(&self, seq_len: usize) -> Result<Array2<f32>> {
        Ok(Array2::ones((1, seq_len)))
    }

    /// Full forward from tokens: embed -> embed_norm -> layers -> final_norm
    fn forward_tokens(
        &self,
        input_ids: &Array2<u32>,
        attention_mask: Option<&Array2<f32>>,
        token_type_ids: Option<&Array2<u32>>,
        pos: usize,
    ) -> Result<CpuEncoderOutput> {
        let hidden: Array3<f32> = self.embed_tokens(input_ids, token_type_ids, pos)?;

        // normalize
        let normalized = self.encoder().embed_norm(&hidden)?;

        let mask = match attention_mask {
            Some(m) => m.clone(),
            None => self.get_attention_mask(normalized.shape()[1])?,
        };

        self.encoder().forward(&normalized, &mask)
    }

    /// Forward from mel spectrogram (Whisper)
    fn forward_audio(
        &self,
        mel: &Array3<f32>, // [batch, n_mels, frames]
        attention_mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let hidden = self.embed_audio(mel)?;

        let mask = match attention_mask {
            Some(m) => m.clone(),
            None => self.get_attention_mask(hidden.shape()[1])?,
        };

        let output =
            self.encoder()
                .forward_layers(&hidden, &mask, 0, self.encoder().num_layers())?;
        self.encoder().final_norm(&output)
    }
}

/// Output from GPU encoder.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
#[derive(Debug)]
pub struct GpuEncoderOutput {
    /// Final hidden states on GPU: `[batch_size, sequence_length, hidden_size]`
    pub last_hidden_state: GpuTensor,
}

/// Placeholder GPU encoder for wasm builds.
///
/// There is no GPU backend on wasm, but the encoder pipeline still carries
/// `Option<Box<dyn GpuEncoder>>` fields and the `EncoderModelFactory` trait still
/// mentions the type in its signature. Declaring an empty trait keeps every one of
/// those signatures compiling against a value that can only ever be `None`, which
/// is far less invasive than threading `#[cfg]` through the pipeline. Nothing can
/// implement it usefully, so no GPU path can be reached by accident.
#[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
pub trait GpuEncoder: Send + Sync {}

/// Placeholder GPU encoder ops for wasm builds. See [`GpuEncoder`].
#[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
pub trait GpuEncoderOps: Send + Sync {}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
pub trait GpuEncoder: Send + Sync {
    /// Compute embeddings only
    fn embed(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
    ) -> Result<GpuTensor>;

    /// Compute embeddings + initial normalization
    fn embed_and_normalize(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
    ) -> Result<GpuTensor>;

    /// Run layers `[start_layer, end_layer)` on hidden states
    fn forward_layers(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor>;

    /// Apply embedding layer normalization (BART-specific, after embed lookup)
    fn embed_norm(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor>;

    /// Apply final layer normalization (T5/Whisper have this, BART doesn't)
    fn final_norm(
        &self,
        _cmd_encoder: &mut wgpu::CommandEncoder,
        _pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        // Default: no final norm (BART)
        Ok(hidden_states.clone())
    }

    /// Number of encoder layers in this model.
    fn num_layers(&self) -> usize;

    /// Hidden dimension of the model.
    fn hidden_size(&self) -> usize;

    /// Full forward pass through the encoder
    #[deprecated]
    fn forward(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        attention_mask: &GpuTensor,
        token_type_ids: Option<ModelInput<'_>>,
    ) -> Result<GpuEncoderOutput> {
        let hidden = self.embed_and_normalize(cmd_encoder, pool, input, token_type_ids)?;
        let output = self.forward_layers(
            cmd_encoder,
            pool,
            &hidden,
            attention_mask,
            0,
            self.num_layers(),
        )?;
        Ok(GpuEncoderOutput {
            last_hidden_state: output,
        })
    }
    fn forward2(
        // todo do something
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
    ) -> Result<GpuEncoderOutput> {
        let output = self.forward_layers(
            cmd_encoder,
            pool,
            hidden_states,
            attention_mask,
            0,
            self.num_layers(),
        )?;
        let normalized = self.final_norm(cmd_encoder, pool, &output)?;
        Ok(GpuEncoderOutput {
            last_hidden_state: normalized,
        })
    }
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
pub trait GpuEncoderOps: Send + Sync {
    /// Access the underlying encoder (transformer layers)
    fn encoder(&self) -> &dyn GpuEncoder;

    /// Embed tokens to hidden states (no normalization)
    fn embed_tokens(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input_ids: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
        pos: usize,
    ) -> Result<GpuTensor>;

    /// Embed audio -> hidden states (Whisper)
    fn embed_audio(
        &self,
        _cmd_encoder: &mut wgpu::CommandEncoder,
        _pool: &mut GpuTensorPool,
        _mel: &GpuTensor,
    ) -> Result<GpuTensor> {
        Err(anyhow!("Audio embedding not supported for this model"))
    }

    fn get_attention_mask(&self, ctx: &Arc<WgpuContext>, seq_len: usize) -> Result<GpuTensor> {
        let mask_cpu = Array2::<f32>::ones((1, seq_len));
        GpuTensor::from_ndarray(ctx, &mask_cpu)
    }

    /// Full forward from tokens: embed -> embed_norm -> layers -> final_norm
    fn forward_tokens(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        ctx: &Arc<WgpuContext>,
        input_ids: ModelInput<'_>,
        attention_mask: Option<&GpuTensor>,
        token_type_ids: Option<ModelInput<'_>>,
        pos: usize,
    ) -> Result<GpuEncoderOutput> {
        // Embed tokens
        let hidden = self.embed_tokens(cmd_encoder, pool, input_ids, token_type_ids, pos)?;

        // Apply embedding normalization
        let normalized = self.encoder().embed_norm(cmd_encoder, pool, &hidden)?;

        // Get or create attention mask
        let mask = match attention_mask {
            Some(m) => m.clone(),
            None => self.get_attention_mask(ctx, normalized.shape()[1])?,
        };

        // Run through encoder layers + final norm
        self.encoder()
            .forward2(cmd_encoder, pool, &normalized, &mask)
    }

    /// Forward from mel spectrogram (Whisper)
    fn forward_audio(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        ctx: &Arc<WgpuContext>,
        mel: &GpuTensor,
        attention_mask: Option<&GpuTensor>,
    ) -> Result<GpuEncoderOutput> {
        let hidden = self.embed_audio(cmd_encoder, pool, mel)?;

        let mask = match attention_mask {
            Some(m) => m.clone(),
            None => self.get_attention_mask(ctx, hidden.shape()[1])?,
        };

        self.encoder().forward2(cmd_encoder, pool, &hidden, &mask)
    }
}

pub fn l2_normalize_inplace(embeddings: &mut Array2<f32>) {
    for mut row in embeddings.rows_mut() {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }
}

#[cfg(test)]
mod tests_trait {
    use crate::{Cache, WgpuContext, traits::InferenceModel};

    use super::*;
    use ndarray::Array2;
    use std::sync::Mutex;
    struct MockCpuEncoder {
        hidden_size: usize,
        captured_mask_sum: Mutex<f32>,
    }

    impl MockCpuEncoder {
        fn new(hidden_size: usize) -> Self {
            Self {
                hidden_size,
                captured_mask_sum: Mutex::new(0.0),
            }
        }
    }
    impl CpuTransformerCore for MockCpuEncoder {
        fn num_layers(&self) -> usize {
            1
        }
        fn hidden_size(&self) -> usize {
            self.hidden_size
        }
        fn final_norm(&self, _hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            unimplemented!()
        }
        fn embed_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            Ok(hidden_states.clone())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn num_attention_heads(&self) -> usize {
            8
        }
    }
    impl CpuEncoder for MockCpuEncoder {
        // Real buffers rather than unimplemented!(): every encode now takes the
        // buffered path, so this is on the tested route.
        fn create_buffers(&self, max_batch: usize, max_seq: usize) -> EncoderBuffers {
            EncoderBuffers::new_auto(
                max_batch,
                max_seq,
                self.hidden_size,
                8,
                self.hidden_size * 4,
            )
        }
        fn forward_layers(
            &self,
            hidden_states: &Array3<f32>,
            attention_mask: &Array2<f32>,
            _start_layer: usize,
            _end_layer: usize,
        ) -> Result<Array3<f32>> {
            let mut lock = self.captured_mask_sum.lock().unwrap();
            *lock = attention_mask.sum();
            Ok(hidden_states.clone())
        }
    }

    struct MockModel {
        encoder: MockCpuEncoder,
    }

    // Minimal Trait Impls for MockModel
    impl LanguageModel for MockModel {
        fn vocab_size(&self) -> usize {
            100
        }
        fn hidden_size(&self) -> usize {
            self.encoder.hidden_size
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn num_heads(&self) -> usize {
            1
        }
        fn context_size(&self) -> usize {
            128
        }
        fn tokenizer(&self) -> &tokenizers::Tokenizer {
            unimplemented!()
        }
        fn bos_token_id(&self) -> Option<u32> {
            None
        }
        fn eos_token_id(&self) -> Option<u32> {
            None
        }
        fn pad_token_id(&self) -> Option<u32> {
            None
        }
        fn forced_bos_token_id(&self) -> Option<u32> {
            None
        }
        fn forced_eos_token_id(&self) -> Option<u32> {
            None
        }
        fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
            unimplemented!()
        }
    }

    impl crate::traits::InferenceModel for MockModel {
        fn device(&self) -> crate::traits::Device {
            crate::traits::Device::Cpu
        }
        fn context(&self) -> Option<std::sync::Arc<WgpuContext>> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl CpuEncoderOps for MockModel {
        fn encoder(&self) -> &dyn CpuEncoder {
            &self.encoder
        }
        fn embed_tokens(
            &self,
            input_ids: &Array2<u32>,
            _token_type_ids: Option<&Array2<u32>>,
            _pos: usize,
        ) -> Result<Array3<f32>> {
            Ok(Array3::zeros((
                input_ids.dim().0,
                input_ids.dim().1,
                self.hidden_size(),
            )))
        }
    }

    impl EncoderLanguageModel for MockModel {
        fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
            Some(self)
        }
        fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
            None
        }
    }
    struct MockGoldenEncoder {
        hidden_states: Array3<f32>,
    }

    impl LanguageModel for MockGoldenEncoder {
        fn vocab_size(&self) -> usize {
            0
        }
        fn hidden_size(&self) -> usize {
            4
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn num_heads(&self) -> usize {
            1
        }
        fn context_size(&self) -> usize {
            5
        }
        fn tokenizer(&self) -> &tokenizers::Tokenizer {
            unimplemented!()
        }
        fn bos_token_id(&self) -> Option<u32> {
            None
        }
        fn eos_token_id(&self) -> Option<u32> {
            None
        }
        fn pad_token_id(&self) -> Option<u32> {
            None
        }
        fn forced_bos_token_id(&self) -> Option<u32> {
            None
        }
        fn forced_eos_token_id(&self) -> Option<u32> {
            None
        }
        fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
            unimplemented!()
        }
    }

    impl InferenceModel for MockGoldenEncoder {
        fn device(&self) -> crate::traits::Device {
            crate::traits::Device::Cpu
        }
        fn context(&self) -> Option<std::sync::Arc<WgpuContext>> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[async_trait]
    impl EncoderLanguageModel for MockGoldenEncoder {
        fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
            None
        }
        fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
            None
        }

        async fn get_hidden_states_batch(
            &self,
            _texts: &[&str],
        ) -> Result<(Array3<f32>, Array2<f32>)> {
            let mask = Array2::from_shape_vec(
                (2, 5),
                vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            )
            .unwrap();
            Ok((self.hidden_states.clone(), mask))
        }
    }

    #[tokio::test]
    async fn test_trait_dispatch_and_mask_conversion() -> Result<()> {
        let hidden_size = 8;
        let model = MockModel {
            encoder: MockCpuEncoder::new(hidden_size),
        };

        let input_ids = Array2::from_elem((2, 3), 1u32);
        let attention_mask_u32 = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 1, 0, 0])?;

        let (output, attention_mask_f32) = model
            .get_hidden_states_batch_from_ids(&input_ids, &attention_mask_u32)
            .await?;

        assert_eq!(output.dim(), (2, 3, hidden_size));
        assert_eq!(attention_mask_f32[[0, 0]], 1.0);
        assert_eq!(attention_mask_f32[[0, 2]], 0.0);
        assert_eq!(attention_mask_f32.sum(), 3.0);

        let captured_sum = *model.encoder.captured_mask_sum.lock().unwrap();
        assert_eq!(captured_sum, 3.0);

        Ok(())
    }

    #[test]
    fn test_l2_normalize_inplace() {
        let mut data = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 1.0, 1.0]).unwrap();
        l2_normalize_inplace(&mut data);

        assert!((data[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((data[[0, 1]] - 0.8).abs() < 1e-6);

        let val = 1.0 / 2.0f32.sqrt();
        assert!((data[[1, 0]] - val).abs() < 1e-6);
        assert!((data[[1, 1]] - val).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_pooling_strategies_golden() -> Result<()> {
        let hidden_states_data = vec![
            -1.331580, -0.437194, 0.457193, 1.351581, -1.331581, -0.437193, 0.457194, 1.351580,
            -1.331581, -0.437194, 0.457194, 1.351581, -1.331581, -0.437194, 0.457194, 1.351581,
            -1.331581, -0.437193, 0.457194, 1.351580, -1.331580, -0.437194, 0.457193, 1.351581,
            -1.331581, -0.437193, 0.457193, 1.351581, -1.331581, -0.437193, 0.457194, 1.351580,
            -1.331581, -0.437193, 0.457193, 1.351581, -1.331581, -0.437193, 0.457193, 1.351581,
        ];
        let hidden = Array3::from_shape_vec((2, 5, 4), hidden_states_data)?;
        let model = MockGoldenEncoder {
            hidden_states: hidden,
        };

        let config_mean = EncodingConfig {
            pooling_strategy: PoolingStrategy::Mean,
            normalize: false,
        };
        let out_mean = model.encode_batch(&["a", "b"], &config_mean).await?;

        let pool_mean_data = [
            -1.331581, -0.437193, 0.457194, 1.351580, -1.331581, -0.437193, 0.457193, 1.351581,
        ];
        for (i, row) in out_mean.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                let golden = pool_mean_data[i * 4 + j];
                assert!(
                    (val - golden).abs() < 1e-5,
                    "Mean Pooling: Got {}, Expected {}",
                    val,
                    golden
                );
            }
        }

        let config_cls = EncodingConfig {
            pooling_strategy: PoolingStrategy::Cls,
            normalize: false,
        };
        let out_cls = model.encode_batch(&["a", "b"], &config_cls).await?;

        let pool_cls_data = [
            -1.331580, -0.437194, 0.457193, 1.351581, -1.331580, -0.437194, 0.457193, 1.351581,
        ];
        for (i, row) in out_cls.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                let golden = pool_cls_data[i * 4 + j];
                assert!((val - golden).abs() < 1e-5, "CLS Pooling mismatch");
            }
        }

        let config_max = EncodingConfig {
            pooling_strategy: PoolingStrategy::Max,
            normalize: false,
        };
        let out_max = model.encode_batch(&["a", "b"], &config_max).await?;

        let pool_max_data = [
            -1.331580, -0.437193, 0.457194, 1.351581, -1.331580, -0.437193, 0.457194, 1.351581,
        ];
        for (i, row) in out_max.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                let golden = pool_max_data[i * 4 + j];
                assert!((val - golden).abs() < 1e-5, "Max Pooling mismatch");
            }
        }

        let config_last = EncodingConfig {
            pooling_strategy: PoolingStrategy::LastToken,
            normalize: false,
        };
        let out_last = model.encode_batch(&["a", "b"], &config_last).await?;

        let pool_last_data = [
            -1.331581, -0.437193, 0.457194, 1.351580, -1.331581, -0.437193, 0.457194, 1.351580,
        ];
        for (i, row) in out_last.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                let golden = pool_last_data[i * 4 + j];
                assert!((val - golden).abs() < 1e-5, "Last Token Pooling mismatch");
            }
        }

        let config_norm = EncodingConfig {
            pooling_strategy: PoolingStrategy::Mean,
            normalize: true,
        };
        let out_norm = model.encode_batch(&["a", "b"], &config_norm).await?;

        let normed_mean_data = [
            -0.665787, -0.218596, 0.228596, 0.675787, -0.665787, -0.218595, 0.228595, 0.675787,
        ];
        for (i, row) in out_norm.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                let golden = normed_mean_data[i * 4 + j];
                assert!((val - golden).abs() < 1e-5, "Normalization mismatch");
            }
        }

        Ok(())
    }
}
