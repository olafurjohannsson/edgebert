//! Activation functions for transformers

use ndarray::{Array3, Array4, Axis};

#[cfg(not(target_arch = "wasm32"))]
use ndarray::parallel::prelude::*;

const SQRT_2_OVER_PI: f32 = 0.7978845608_f32;
const GELU_COEFF: f32 = 0.044715_f32;

/// Activation function types
pub enum Activation {
    Gelu,
    Relu,
    Tanh,
    Swish,
}

/// Apply GELU activation in-place
#[inline(always)]
pub fn gelu(x: &mut Array3<f32>) {
    let scaling_factor = (2.0f32).sqrt() / 2.0;
    #[cfg(not(target_arch = "wasm32"))]
    {
        // x.par_mapv_inplace(|val| {
        //     let val_squared = val * val;
        //     let val_cubed = val_squared * val;
        //     let inner = SQRT_2_OVER_PI * (val + GELU_COEFF * val_cubed);
        //     val * 0.5 * (1.0 + inner.tanh())
        // });
        const SQRT_2_OVER_PI: f32 = 0.7978845608_f32;
        const GELU_COEFF: f32 = 0.044715_f32;

        // This implementation now runs on ALL targets (wasm and native).
        x.mapv_inplace(|val| {
            let inner = SQRT_2_OVER_PI * (val + GELU_COEFF * val.powi(3));
            0.5 * val * (1.0 + inner.tanh())
        });
        // const SCALING_FACTOR: f32 = 0.7071067811865475;

        // // The formula used by PyTorch's default GELU is:
        // // 0.5 * x * (1 + erf(x / sqrt(2)))
        // // which is equivalent to:
        // // 0.5 * x * (1 + erf(x * SCALING_FACTOR))
        // // Note: erff is the single-precision (f32) version of the error function.

        // // This code works for both wasm and native builds.
        // x.mapv_inplace(|val| 0.5 * val * (1.0 + erff(val * SCALING_FACTOR)));
    }

    #[cfg(target_arch = "wasm32")]
    {
        x.mapv_inplace(|val| {
            let val_squared = val * val;
            let val_cubed = val_squared * val;
            let inner = SQRT_2_OVER_PI * (val + GELU_COEFF * val_cubed);
            val * 0.5 * (1.0 + inner.tanh())
        });
    }
}

/// Compute softmax over the last dimension of a 4D tensor
#[inline(always)]
pub fn softmax(scores: &Array4<f32>) -> Array4<f32> {
    // Find max for numerical stability
    let max_vals = scores.fold_axis(Axis(3), f32::NEG_INFINITY, |&acc, &x| acc.max(x));
    let max_expanded = max_vals.insert_axis(Axis(3));

    // Compute exp(x - max)
    let mut result = scores - &max_expanded;
    result.mapv_inplace(f32::exp);

    // Normalize
    let sum_exp = result.sum_axis(Axis(3)).insert_axis(Axis(3));
    result /= &sum_exp;

    result
}

/// Apply ReLU activation
pub fn relu(x: &mut Array3<f32>) {
    x.mapv_inplace(|val| val.max(0.0));
}

/// Apply Swish/SiLU activation
pub fn swish(x: &mut Array3<f32>) {
    x.mapv_inplace(|val| val * (1.0 / (1.0 + (-val).exp())));
}
