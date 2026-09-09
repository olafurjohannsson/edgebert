//! Pooling traits and implementations for encoder outputs

use super::config::PoolingStrategy;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::{cls_pool, last_token_pool, max_pool, mean_pool};
use anyhow::Result;
use ndarray::{Array2, Array3};

/// CPU pooling head - reduces sequence to single vector.
pub trait CpuPooler: Send + Sync {
    /// Pool hidden states to a single vector per batch item.
    fn pool(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        strategy: &PoolingStrategy,
    ) -> Result<Array2<f32>>;
}

/// GPU pooling head - reduces sequence to single vector.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
pub trait GpuPooler: Send + Sync {
    /// Pool hidden states to a single vector per batch item.
    fn pool(
        &self,
        cmd: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        strategy: &PoolingStrategy,
    ) -> Result<GpuTensor>;
}

/// Standard CPU pooler implementation.
pub struct StandardCpuPooler;

impl CpuPooler for StandardCpuPooler {
    fn pool(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        strategy: &PoolingStrategy,
    ) -> Result<Array2<f32>> {
        match strategy {
            PoolingStrategy::Cls => cls_pool(hidden_states),
            PoolingStrategy::Mean => mean_pool(hidden_states, attention_mask),
            PoolingStrategy::Max => max_pool(hidden_states, attention_mask),
            PoolingStrategy::LastToken => last_token_pool(hidden_states, attention_mask),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cpu_pooler_cls() {
        let hidden = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let mask = array![[1.0, 1.0], [1.0, 0.0]];
        let pooler = StandardCpuPooler;

        let pooled = pooler.pool(&hidden, &mask, &PoolingStrategy::Cls).unwrap();
        assert_eq!(pooled[[0, 0]], 1.0);
        assert_eq!(pooled[[1, 1]], 6.0);
    }

    #[test]
    fn test_cpu_pooler_mean() {
        let hidden = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let mask = array![[1.0, 1.0], [1.0, 0.0]];
        let pooler = StandardCpuPooler;

        let pooled = pooler.pool(&hidden, &mask, &PoolingStrategy::Mean).unwrap();
        assert!((pooled[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((pooled[[1, 1]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_pooler_max() {
        let hidden = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let mask = array![[1.0, 1.0], [1.0, 0.0]];
        let pooler = StandardCpuPooler;

        let pooled = pooler.pool(&hidden, &mask, &PoolingStrategy::Max).unwrap();
        assert_eq!(pooled[[0, 0]], 3.0);
        assert_eq!(pooled[[1, 1]], 6.0);
    }

    #[test]
    fn test_cpu_pooler_last_token() {
        let hidden = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];
        let mask = array![[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
        let pooler = StandardCpuPooler;

        let pooled = pooler
            .pool(&hidden, &mask, &PoolingStrategy::LastToken)
            .unwrap();
        assert_eq!(pooled[[0, 0]], 3.0);
        assert_eq!(pooled[[1, 1]], 12.0);
    }
}
