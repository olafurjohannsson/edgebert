//! CPU-based decoder generation backend using ndarray

use crate::cache::Cache;
use crate::decoder::prelude::*;
use crate::models::base::AutoregressiveLoop;
use crate::utils::time::Instant;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use log::{debug, trace};
use ndarray::{Array1, Array2, Array3, s};

/// CPU-based backend for autoregressive decoder generation.
#[derive(Clone, Default)]
pub struct CpuDecoderBackend;

impl CpuDecoderBackend {
    /// Creates a new CPU decoder backend.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DecoderGenerationBackend for CpuDecoderBackend {
    /// Token tensor type for decode loop: 2D ndarray with shape `[1, 1]`.
    type DecodeToken = Array2<u32>;

    /// Creates a single-token tensor for the decode loop.
    fn new_decode_token(&self) -> Result<Self::DecodeToken> {
        Ok(Array2::zeros((1, 1)))
    }

    /// Updates the decode token tensor with a newly sampled token ID.
    fn update_decode_token(&self, tensor: &mut Self::DecodeToken, new_token_id: u32) -> Result<()> {
        tensor[[0, 0]] = new_token_id;
        Ok(())
    }

    /// Processes the prompt and returns logits for the first generated token.
    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        tokens: &Array2<u32>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if tokens.is_empty() {
            return Err(anyhow!("Cannot prefill with empty prompt"));
        }

        let ops = model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        let prompt_len = tokens.shape()[1];
        let prefill_start = Instant::now();

        debug!(
            "CPU prefill: {} tokens, strategy={:?}",
            prompt_len,
            model.autoregressive_loop()
        );

        let logits = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => self.prefill_pipelined(ops, tokens, cache)?,
            AutoregressiveLoop::Legacy => self.prefill_legacy(ops, tokens, cache)?,
        };

        debug!(
            "CPU prefill complete: {:.2}ms",
            prefill_start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(logits)
    }

    /// Prefill continuing from a warm cache. `start_pos` tokens are already in it.
    async fn prefill_at(
        &self,
        model: &dyn DecoderLanguageModel,
        tokens: &Array2<u32>,
        start_pos: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if start_pos == 0 {
            return self.prefill(model, tokens, cache).await;
        }
        if tokens.is_empty() {
            return Err(anyhow!("Cannot prefill with empty prompt"));
        }
        let ops = model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => {
                self.prefill_pipelined_from(ops, tokens, start_pos, cache)
            }
            // The legacy loop has different cache semantics and no offset path;
            // reusing a prefix there would silently mis-position the new tokens.
            AutoregressiveLoop::Legacy => Err(anyhow!(
                "prefix reuse is not supported for the legacy autoregressive loop"
            )),
        }
    }

    /// Processes a single token and returns logits for the next token.
    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        token_tensor: &Self::DecodeToken,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let ops = model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        // Position offset is sequence length minus 1 (0-indexed)
        let past_len = seq_len - 1;

        // Embed the single token
        let hidden_states: Array3<f32> = ops.embed(token_tensor, past_len)?;

        // Build attention mask
        let mask_len = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => seq_len,
            AutoregressiveLoop::Legacy => seq_len + 1,
        };

        let attention_mask = ops.get_attention_mask(1, mask_len - 1)?;

        trace!(
            "CPU decode: token={}, seq_len={}, mask_len={}, past_len={}",
            token_tensor[[0, 0]],
            seq_len,
            mask_len,
            past_len
        );

        // Forward pass through decoder layers (without final norm)
        let decoder_output =
            ops.decoder()
                .forward(&hidden_states, &attention_mask, past_len, Some(cache))?;

        // Project to logits and extract single position
        let logits_3d = ops.project_to_logits(&decoder_output)?;

        // Shape: [1, 1, vocab_size] -> [vocab_size]
        Ok(logits_3d.slice(s![0, 0, ..]).to_owned())
    }
}

impl CpuDecoderBackend {
    /// Pipelined prefill
    fn prefill_pipelined(
        &self,
        ops: &dyn CpuDecoderOps,
        tokens: &Array2<u32>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let prompt_len = tokens.shape()[1];

        // Build causal mask for full sequence
        let attention_mask = ops.get_attention_mask(prompt_len, 0)?;

        // Embed all tokens
        let hidden_states = ops.embed(tokens, 0)?;

        // Single forward pass through all layers (includes final_norm)
        let decoder_output = ops.decoder().forward(
            &hidden_states,
            &attention_mask,
            0, // Position offset is 0 for prefill
            Some(cache),
        )?;

        // Project to logits
        let logits_3d = ops.project_to_logits(&decoder_output)?;

        // Extract last position: [1, seq, vocab] -> [vocab]
        Ok(logits_3d.slice(s![0, -1, ..]).to_owned())
    }

    /// Pipelined prefill that continues from a warm cache.
    ///
    /// `start_pos` is how many tokens the cache already holds. The attention
    /// already handles a non-zero position offset — `start_write` is derived from
    /// the cache length — so this is the same code path with the offset threaded
    /// through rather than hardcoded to zero. Verified against a whole-prompt
    /// prefill: identical logits, same argmax.
    fn prefill_pipelined_from(
        &self,
        ops: &dyn CpuDecoderOps,
        tokens: &Array2<u32>,
        start_pos: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let new_len = tokens.shape()[1];
        let attention_mask = ops.get_attention_mask(new_len, start_pos)?;
        let hidden_states = ops.embed(tokens, start_pos)?;

        let decoder_output =
            ops.decoder()
                .forward(&hidden_states, &attention_mask, start_pos, Some(cache))?;

        let logits_3d = ops.project_to_logits(&decoder_output)?;
        Ok(logits_3d.slice(s![0, -1, ..]).to_owned())
    }

    /// Legacy prefill
    ///
    /// This path is kept for backward compatibility with older models
    /// that have different cache semantics.
    fn prefill_legacy(
        &self,
        ops: &dyn CpuDecoderOps,
        tokens: &Array2<u32>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let prompt_len = tokens.shape()[1];

        let hidden_states = ops.embed(tokens, 0)?;
        let mask_full = Array2::ones((1, prompt_len));
        ops.decoder()
            .forward(&hidden_states, &mask_full, 0, Some(cache))?;
        let last_token = tokens[[0, prompt_len - 1]];
        let last_token_array = Array2::from_elem((1, 1), last_token);
        let hidden_states = ops.embed(&last_token_array, prompt_len - 1)?;
        let mask_step = ops.get_attention_mask(1, prompt_len)?;
        let decoder_output =
            ops.decoder()
                .forward(&hidden_states, &mask_step, prompt_len, Some(cache))?;
        let logits_3d = ops.project_to_logits(&decoder_output)?;
        Ok(logits_3d.slice(s![0, 0, ..]).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_decode_token_shape() {
        let backend = CpuDecoderBackend::new();
        let token = backend.new_decode_token().unwrap();

        assert_eq!(token.shape(), &[1, 1]);
        assert_eq!(token[[0, 0]], 0);
    }

    #[test]
    fn test_update_decode_token() {
        let backend = CpuDecoderBackend::new();
        let mut token = backend.new_decode_token().unwrap();

        backend.update_decode_token(&mut token, 42).unwrap();
        assert_eq!(token[[0, 0]], 42);

        backend.update_decode_token(&mut token, 1337).unwrap();
        assert_eq!(token[[0, 0]], 1337);
    }

    #[test]
    fn test_backend_is_stateless() {
        // Backend should be Clone and Default since it's stateless
        let backend1 = CpuDecoderBackend;
        let backend2 = backend1.clone();

        // Both should work independently
        let token1 = backend1.new_decode_token().unwrap();
        let token2 = backend2.new_decode_token().unwrap();

        assert_eq!(token1, token2);
    }

    #[test]
    fn test_token_array_shape() {
        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();

        assert_eq!(tokens.shape(), &[1, 5]);
        assert_eq!(tokens.shape()[1], 5); // seq_len
    }
}
