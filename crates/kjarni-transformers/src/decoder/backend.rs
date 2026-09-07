//! Type-erased backend that dispatches to CPU or GPU implementations.

use std::any::Any;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array1, Array2};

use crate::cache::Cache;
use crate::decoder::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::GpuTensor;

#[derive(Clone)]
pub enum AnyDecoderBackend {
    Cpu(CpuDecoderBackend),
    // No GPU backend exists on wasm, so the variant does not either. Every match on
    // this enum is exhaustive on both targets because the arm is gated too.
    #[cfg(not(target_arch = "wasm32"))]
    Gpu(Arc<GpuDecoderBackend>),
}

impl AnyDecoderBackend {
    pub fn cpu() -> Self {
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu(backend: Arc<GpuDecoderBackend>) -> Self {
        AnyDecoderBackend::Gpu(backend)
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, AnyDecoderBackend::Cpu(_))
    }

    pub fn is_gpu(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return matches!(self, AnyDecoderBackend::Gpu(_));
        #[cfg(target_arch = "wasm32")]
        return false;
    }

    pub fn backend_type(&self) -> &'static str {
        match self {
            AnyDecoderBackend::Cpu(_) => "CPU",
            #[cfg(not(target_arch = "wasm32"))]
            AnyDecoderBackend::Gpu(_) => "GPU",
        }
    }
}

#[async_trait]
impl DecoderGenerationBackend for AnyDecoderBackend {
    type DecodeToken = Box<dyn Any + Send + Sync>;

    fn new_decode_token(&self) -> Result<Self::DecodeToken> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let token = backend.new_decode_token()?;
                Ok(Box::new(token))
            }
            #[cfg(not(target_arch = "wasm32"))]
            AnyDecoderBackend::Gpu(backend) => {
                let token = backend.new_decode_token()?;
                Ok(Box::new(token))
            }
        }
    }

    fn update_decode_token(&self, token: &mut Self::DecodeToken, new_token_id: u32) -> Result<()> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let concrete = token
                    .downcast_mut::<Array2<u32>>()
                    .ok_or_else(|| anyhow!("cpu backend expected Array2<u32>, got wrong type"))?;
                backend.update_decode_token(concrete, new_token_id)
            }
            #[cfg(not(target_arch = "wasm32"))]
            AnyDecoderBackend::Gpu(backend) => {
                let concrete = token
                    .downcast_mut::<GpuTensor>()
                    .ok_or_else(|| anyhow!("gpu backend expected GpuTensor, got wrong type"))?;
                backend.update_decode_token(concrete, new_token_id)
            }
        }
    }

    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        tokens: &Array2<u32>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => backend.prefill(model, tokens, cache).await,
            #[cfg(not(target_arch = "wasm32"))]
            AnyDecoderBackend::Gpu(backend) => backend.prefill(model, tokens, cache).await,
        }
    }

    /// Prefill continuing from a warm cache. Backends that cannot do this fall
    /// back to a full prefill, which is correct but saves nothing.
    async fn prefill_at(
        &self,
        model: &dyn DecoderLanguageModel,
        tokens: &Array2<u32>,
        start_pos: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                backend.prefill_at(model, tokens, start_pos, cache).await
            }
            #[cfg(not(target_arch = "wasm32"))]
            AnyDecoderBackend::Gpu(backend) => {
                backend.prefill_at(model, tokens, start_pos, cache).await
            }
        }
    }

    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        token: &Self::DecodeToken,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let concrete = token
                    .downcast_ref::<Array2<u32>>()
                    .ok_or_else(|| anyhow!("cpu backend expected Array2<u32>, got wrong type"))?;
                backend.decode_one(model, concrete, seq_len, cache).await
            }
            #[cfg(not(target_arch = "wasm32"))]
            AnyDecoderBackend::Gpu(backend) => {
                let concrete = token
                    .downcast_ref::<GpuTensor>()
                    .ok_or_else(|| anyhow!("gpu backend expected GpuTensor, got wrong type"))?;
                backend.decode_one(model, concrete, seq_len, cache).await
            }
        }
    }
}

impl std::fmt::Debug for AnyDecoderBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnyDecoderBackend::Cpu(_) => write!(f, "AnyDecoderBackend::Cpu"),
            #[cfg(not(target_arch = "wasm32"))]
            AnyDecoderBackend::Gpu(_) => write!(f, "AnyDecoderBackend::Gpu"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::decoder::CpuDecoderBackend;

    #[test]
    fn test_any_backend_cpu_constructor() {
        let backend = AnyDecoderBackend::cpu();
        assert!(backend.is_cpu());
        assert!(!backend.is_gpu());
        assert_eq!(backend.backend_type(), "CPU");
    }

    #[test]
    fn test_any_backend_is_cpu() {
        let cpu = CpuDecoderBackend::new();
        let any = AnyDecoderBackend::Cpu(cpu);

        assert!(any.is_cpu());
        assert!(!any.is_gpu());
    }

    #[test]
    fn test_any_backend_new_decode_token_cpu() {
        let any = AnyDecoderBackend::cpu();

        let token = any.new_decode_token();
        assert!(token.is_ok());

        let t = token.unwrap();
        let concrete = t.downcast_ref::<Array2<u32>>();
        assert!(concrete.is_some());
        assert_eq!(concrete.unwrap().shape(), &[1, 1]);
    }

    #[test]
    fn test_any_backend_update_decode_token_cpu() {
        let any = AnyDecoderBackend::cpu();

        let mut token = any.new_decode_token().unwrap();
        let update = any.update_decode_token(&mut token, 123);
        assert!(update.is_ok());

        let concrete = token.downcast_ref::<Array2<u32>>().unwrap();
        assert_eq!(concrete[[0, 0]], 123);
    }

    #[test]
    fn test_any_backend_update_decode_token_multiple_times() {
        let any = AnyDecoderBackend::cpu();
        let mut token = any.new_decode_token().unwrap();

        for i in 0..100 {
            any.update_decode_token(&mut token, i).unwrap();
            let concrete = token.downcast_ref::<Array2<u32>>().unwrap();
            assert_eq!(concrete[[0, 0]], i);
        }
    }

    #[test]
    fn test_decode_token_type_mismatch_error() {
        let any = AnyDecoderBackend::cpu();

        let mut fake_token: Box<dyn Any + Send + Sync> = Box::new(String::from("fake"));

        let result = any.update_decode_token(&mut fake_token, 1);
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("cpu backend expected"));
    }

    #[test]
    fn test_any_backend_debug_impl() {
        let cpu = AnyDecoderBackend::cpu();
        assert_eq!(format!("{:?}", cpu), "AnyDecoderBackend::Cpu");
    }

    #[test]
    fn test_any_backend_clone() {
        let backend1 = AnyDecoderBackend::cpu();
        let backend2 = backend1.clone();

        assert!(backend1.is_cpu());
        assert!(backend2.is_cpu());

        let token1 = backend1.new_decode_token().unwrap();
        let token2 = backend2.new_decode_token().unwrap();

        let concrete1 = token1.downcast_ref::<Array2<u32>>().unwrap();
        let concrete2 = token2.downcast_ref::<Array2<u32>>().unwrap();
        assert_eq!(concrete1, concrete2);
    }
}
