//! Activation functions and softmax operations.

use std::str::FromStr;

use libm::{erff, expf, tanhf};
use ndarray::{
    Array1, Array2, Array3, Array4, ArrayBase, ArrayViewMut2, DataMut, parallel::prelude::*, s,
};
use serde::{Deserialize, Serialize};

/// Minimum array size for parallel execution.
pub const PARALLEL_THRESHOLD: usize = 16_384;

const SQRT_2_INV: f32 = 0.7071067811865475;
const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

/// Supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum Activation {
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    #[default]
    GeluNew,
    #[serde(alias = "relu")]
    Relu,
    #[serde(alias = "silu", alias = "swish")]
    SilU,
    #[serde(alias = "tanh")]
    Tanh,
    /// `x * sigmoid(1.702x)`, the sigmoid-weighted approximation OpenAI shipped
    /// with CLIP. Not interchangeable with the tanh form: they differ by enough
    /// to move embeddings, so a checkpoint declaring `quick_gelu` needs this one.
    #[serde(alias = "quick_gelu")]
    QuickGelu,
}

impl FromStr for Activation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gelu" => Ok(Activation::GeluNew),
            "gelu_new" | "gelu_fast" => Ok(Activation::GeluNew),
            "relu" => Ok(Activation::Relu),
            "silu" | "swish" => Ok(Activation::SilU),
            "tanh" => Ok(Activation::Tanh),
            "quick_gelu" | "quickgelu" => Ok(Activation::QuickGelu),
            _ => Err(format!("unknown activation function: {}", s)),
        }
    }
}

#[inline(always)]
pub fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + erff(x * SQRT_2_INV))
}

#[inline(always)]
pub fn quick_gelu_scalar(x: f32) -> f32 {
    x / (1.0 + (-1.702 * x).exp())
}

pub fn gelu_new_scalar(x: f32) -> f32 {
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    0.5 * x * (1.0 + tanhf(inner))
}

#[inline(always)]
pub fn relu_scalar(x: f32) -> f32 {
    x.max(0.0)
}

#[inline(always)]
pub fn silu_scalar(x: f32) -> f32 {
    if x <= -20.0 {
        0.0
    } else if x >= 20.0 {
        x
    } else {
        x / (1.0 + expf(-x))
    }
}

#[inline(always)]
pub fn silu_fast_scalar(x: f32) -> f32 {
    x / (1.0 + expf(-x))
}

#[inline(always)]
pub fn tanh_scalar(x: f32) -> f32 {
    tanhf(x)
}

fn apply_activation_slice(slice: &mut [f32], activation: Activation, use_parallel: bool) {
    match (activation, use_parallel) {
        (Activation::Gelu, true) => slice.par_iter_mut().for_each(|x| *x = gelu_scalar(*x)),
        (Activation::Gelu, false) => slice.iter_mut().for_each(|x| *x = gelu_scalar(*x)),
        (Activation::GeluNew, true) => slice.par_iter_mut().for_each(|x| *x = gelu_new_scalar(*x)),
        (Activation::GeluNew, false) => slice.iter_mut().for_each(|x| *x = gelu_new_scalar(*x)),
        (Activation::Relu, true) => slice.par_iter_mut().for_each(|x| *x = relu_scalar(*x)),
        (Activation::Relu, false) => slice.iter_mut().for_each(|x| *x = relu_scalar(*x)),
        (Activation::SilU, true) => slice.par_iter_mut().for_each(|x| *x = silu_scalar(*x)),
        (Activation::SilU, false) => slice.iter_mut().for_each(|x| *x = silu_scalar(*x)),
        (Activation::Tanh, true) => slice.par_iter_mut().for_each(|x| *x = tanh_scalar(*x)),
        (Activation::Tanh, false) => slice.iter_mut().for_each(|x| *x = tanh_scalar(*x)),
        (Activation::QuickGelu, true) => slice
            .par_iter_mut()
            .for_each(|x| *x = quick_gelu_scalar(*x)),
        (Activation::QuickGelu, false) => slice.iter_mut().for_each(|x| *x = quick_gelu_scalar(*x)),
    }
}

/// Applies activation in-place to a 2D array view.
pub fn apply_activation_2d_mut(arr: &mut ArrayViewMut2<f32>, activation: Activation) {
    let use_parallel = arr.len() >= PARALLEL_THRESHOLD;
    if let Some(slice) = arr.as_slice_mut() {
        apply_activation_slice(slice, activation, use_parallel);
    } else {
        match (activation, use_parallel) {
            (Activation::Gelu, true) => arr.par_mapv_inplace(gelu_scalar),
            (Activation::Gelu, false) => arr.mapv_inplace(gelu_scalar),
            (Activation::GeluNew, true) => arr.par_mapv_inplace(gelu_new_scalar),
            (Activation::GeluNew, false) => arr.mapv_inplace(gelu_new_scalar),
            (Activation::Relu, true) => arr.par_mapv_inplace(relu_scalar),
            (Activation::Relu, false) => arr.mapv_inplace(relu_scalar),
            (Activation::SilU, true) => arr.par_mapv_inplace(silu_scalar),
            (Activation::SilU, false) => arr.mapv_inplace(silu_scalar),
            (Activation::Tanh, true) => arr.par_mapv_inplace(tanh_scalar),
            (Activation::Tanh, false) => arr.mapv_inplace(tanh_scalar),
            (Activation::QuickGelu, true) => arr.par_mapv_inplace(quick_gelu_scalar),
            (Activation::QuickGelu, false) => arr.mapv_inplace(quick_gelu_scalar),
        }
    }
}

/// Applies activation in-place to a 2D array.
pub fn apply_activation_2d(arr: &mut Array2<f32>, activation: Activation) {
    let use_parallel = arr.len() >= PARALLEL_THRESHOLD;
    if let Some(slice) = arr.as_slice_mut() {
        apply_activation_slice(slice, activation, use_parallel);
    } else {
        match (activation, use_parallel) {
            (Activation::Gelu, true) => arr.par_mapv_inplace(gelu_scalar),
            (Activation::Gelu, false) => arr.mapv_inplace(gelu_scalar),
            (Activation::GeluNew, true) => arr.par_mapv_inplace(gelu_new_scalar),
            (Activation::GeluNew, false) => arr.mapv_inplace(gelu_new_scalar),
            (Activation::Relu, true) => arr.par_mapv_inplace(relu_scalar),
            (Activation::Relu, false) => arr.mapv_inplace(relu_scalar),
            (Activation::SilU, true) => arr.par_mapv_inplace(silu_scalar),
            (Activation::SilU, false) => arr.mapv_inplace(silu_scalar),
            (Activation::Tanh, true) => arr.par_mapv_inplace(tanh_scalar),
            (Activation::Tanh, false) => arr.mapv_inplace(tanh_scalar),
            (Activation::QuickGelu, true) => arr.par_mapv_inplace(quick_gelu_scalar),
            (Activation::QuickGelu, false) => arr.mapv_inplace(quick_gelu_scalar),
        }
    }
}

/// Applies activation in-place to a 3D array.
pub fn apply_activation(arr: &mut Array3<f32>, activation: Activation) {
    let use_parallel = arr.len() >= PARALLEL_THRESHOLD;
    if let Some(slice) = arr.as_slice_mut() {
        apply_activation_slice(slice, activation, use_parallel);
    } else {
        match (activation, use_parallel) {
            (Activation::Gelu, true) => arr.par_mapv_inplace(gelu_scalar),
            (Activation::Gelu, false) => arr.mapv_inplace(gelu_scalar),
            (Activation::GeluNew, true) => arr.par_mapv_inplace(gelu_new_scalar),
            (Activation::GeluNew, false) => arr.mapv_inplace(gelu_new_scalar),
            (Activation::Relu, true) => arr.par_mapv_inplace(relu_scalar),
            (Activation::Relu, false) => arr.mapv_inplace(relu_scalar),
            (Activation::SilU, true) => arr.par_mapv_inplace(silu_scalar),
            (Activation::SilU, false) => arr.mapv_inplace(silu_scalar),
            (Activation::Tanh, true) => arr.par_mapv_inplace(tanh_scalar),
            (Activation::Tanh, false) => arr.mapv_inplace(tanh_scalar),
            (Activation::QuickGelu, true) => arr.par_mapv_inplace(quick_gelu_scalar),
            (Activation::QuickGelu, false) => arr.mapv_inplace(quick_gelu_scalar),
        }
    }
}

// Legacy wrappers for backwards compatibility

pub fn gelu(x: &mut Array3<f32>) {
    apply_activation(x, Activation::Gelu);
}

pub fn gelu_new(x: &mut Array3<f32>) {
    apply_activation(x, Activation::GeluNew);
}

pub fn gelu_parallel(x: &mut Array3<f32>) {
    x.par_mapv_inplace(gelu_scalar);
}

pub fn gelu_new_parallel(x: &mut Array3<f32>) {
    x.par_mapv_inplace(gelu_new_scalar);
}

pub fn relu(x: &mut Array3<f32>) {
    apply_activation(x, Activation::Relu);
}

pub fn relu_parallel(x: &mut Array3<f32>) {
    x.par_mapv_inplace(relu_scalar);
}

pub fn silu_generic<S, D>(x: &mut ArrayBase<S, D>)
where
    S: DataMut<Elem = f32>,
    D: ndarray::Dimension,
{
    x.mapv_inplace(silu_scalar);
}

pub fn silu_fast<S, D>(x: &mut ArrayBase<S, D>)
where
    S: DataMut<Elem = f32>,
    D: ndarray::Dimension,
{
    x.mapv_inplace(silu_fast_scalar);
}

pub fn silu_parallel_3d(x: &mut Array3<f32>) {
    x.par_mapv_inplace(silu_scalar);
}

pub fn silu_parallel(x: &mut Array2<f32>) {
    x.par_mapv_inplace(silu_scalar);
}

/// Applies softmax in-place to a slice.
pub fn softmax_inplace(slice: &mut [f32]) {
    if slice.is_empty() {
        return;
    }

    let max = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let mut sum = 0.0;
    for v in slice.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    if sum > 0.0 {
        let scale = 1.0 / sum;
        for v in slice.iter_mut() {
            *v *= scale;
        }
    }
}

/// Applies softmax in-place to a 1D array.
pub fn softmax_1d_inplace(logits: &mut Array1<f32>) {
    if let Some(slice) = logits.as_slice_mut() {
        softmax_inplace(slice);
    } else {
        let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        logits.mapv_inplace(|x| (x - max).exp());
        let sum = logits.sum();
        if sum > 0.0 {
            *logits /= sum;
        }
    }
}

/// Applies softmax along the last axis of a 4D array view.
pub fn softmax_4d_view_inplace(scores: &mut ndarray::ArrayViewMut4<f32>) {
    let (batch_size, num_heads, q_len, _) = scores.dim();

    for b in 0..batch_size {
        for h in 0..num_heads {
            for q in 0..q_len {
                let mut row_view = scores.slice_mut(s![b, h, q, ..]);
                if let Some(slice) = row_view.as_slice_mut() {
                    softmax_inplace(slice);
                } else {
                    let max = row_view.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    row_view.mapv_inplace(|x| (x - max).exp());
                    let sum = row_view.sum();
                    if sum > 0.0 {
                        row_view /= sum;
                    }
                }
            }
        }
    }
}

/// Applies softmax along the last axis of a 4D array.
pub fn softmax_4d_inplace(scores: &mut Array4<f32>) {
    let (batch_size, num_heads, q_len, _) = scores.dim();

    for b in 0..batch_size {
        for h in 0..num_heads {
            for q in 0..q_len {
                let mut row_view = scores.slice_mut(s![b, h, q, ..]);

                if let Some(slice) = row_view.as_slice_mut() {
                    softmax_inplace(slice);
                } else {
                    let max = row_view.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    row_view.mapv_inplace(|x| (x - max).exp());
                    let sum = row_view.sum();
                    if sum > 0.0 {
                        row_view /= sum;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array, arr1};

    #[test]
    fn test_scalars() {
        assert_eq!(relu_scalar(1.0), 1.0);
        assert_eq!(relu_scalar(-1.0), 0.0);

        assert_relative_eq!(silu_scalar(0.0), 0.0);
        assert_relative_eq!(silu_scalar(1.0), 0.7310586, epsilon = 1e-5);
        assert_relative_eq!(silu_scalar(-100.0), 0.0);
        assert_relative_eq!(silu_scalar(100.0), 100.0);

        assert_relative_eq!(tanh_scalar(0.0), 0.0);
        assert_relative_eq!(tanh_scalar(10.0), 1.0, epsilon = 1e-4);

        assert_relative_eq!(gelu_scalar(0.0), 0.0);
        assert_relative_eq!(gelu_scalar(1.0), 0.8413447, epsilon = 1e-5);

        assert_relative_eq!(gelu_new_scalar(0.0), 0.0);
        assert_relative_eq!(gelu_new_scalar(1.0), 0.841192, epsilon = 1e-5);
    }

    #[test]
    fn test_apply_activation_3d() {
        let variants = vec![
            Activation::Gelu,
            Activation::GeluNew,
            Activation::Relu,
            Activation::SilU,
            Activation::Tanh,
        ];

        for act in variants {
            let mut arr = Array3::from_elem((1, 1, 5), -2.0f32);
            apply_activation(&mut arr, act);

            if act == Activation::Relu {
                assert_eq!(arr[[0, 0, 0]], 0.0);
            } else {
                assert_ne!(arr[[0, 0, 0]], -2.0);
            }
        }
    }

    #[test]
    fn test_legacy_wrappers() {
        let mut a = Array3::from_elem((1, 1, 1), 1.0f32);
        gelu(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.8413447, epsilon = 1e-5);

        a.fill(1.0);
        gelu_new(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.841192, epsilon = 1e-5);

        a.fill(-1.0);
        relu(&mut a);
        assert_eq!(a[[0, 0, 0]], 0.0);
    }

    #[test]
    fn test_legacy_parallel_wrappers() {
        let mut a = Array3::from_elem((1, 1, 1), 1.0f32);
        gelu_parallel(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.8413447, epsilon = 1e-5);

        a.fill(1.0);
        gelu_new_parallel(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.841192, epsilon = 1e-5);

        a.fill(-1.0);
        relu_parallel(&mut a);
        assert_eq!(a[[0, 0, 0]], 0.0);

        a.fill(1.0);
        silu_parallel_3d(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.7310586, epsilon = 1e-5);
    }

    #[test]
    fn test_silu_variants() {
        let mut a = Array3::from_elem((1, 1, 1), 1.0f32);
        silu_generic(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.7310586, epsilon = 1e-5);

        a.fill(1.0);
        silu_fast(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.7310586, epsilon = 1e-5);

        let mut b = Array2::from_elem((1, 1), 1.0f32);
        silu_parallel(&mut b);
        assert_relative_eq!(b[[0, 0]], 0.7310586, epsilon = 1e-5);
    }

    #[test]
    fn test_activation_from_str() {
        assert_eq!(Activation::from_str("gelu").unwrap(), Activation::GeluNew);
        assert_eq!(
            Activation::from_str("gelu_new").unwrap(),
            Activation::GeluNew
        );
        assert_eq!(Activation::from_str("relu").unwrap(), Activation::Relu);
        assert_eq!(Activation::from_str("silu").unwrap(), Activation::SilU);
        assert_eq!(Activation::from_str("swish").unwrap(), Activation::SilU);
        assert_eq!(Activation::from_str("tanh").unwrap(), Activation::Tanh);
        assert!(Activation::from_str("invalid").is_err());
    }

    #[test]
    fn test_default() {
        assert_eq!(Activation::default(), Activation::GeluNew);
    }

    #[test]
    fn test_parallel_execution() {
        let size = PARALLEL_THRESHOLD + 100;

        let mut arr = Array3::from_elem((1, 1, size), 1.0f32);
        apply_activation(&mut arr, Activation::Relu);
        assert_eq!(arr[[0, 0, 0]], 1.0);

        let mut arr2 = Array2::from_elem((1, size), 1.0f32);
        apply_activation_2d(&mut arr2, Activation::Relu);
        assert_eq!(arr2[[0, 0]], 1.0);
    }

    #[test]
    fn test_apply_activation_2d_manual() {
        let mut arr = Array2::from_elem((2, 2), -1.0f32);
        apply_activation_2d(&mut arr, Activation::Relu);
        assert!(arr.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_softmax_inplace_basic() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut data);
        assert_relative_eq!(data[0], 0.09003057, epsilon = 1e-6);
        assert_relative_eq!(data[1], 0.24472847, epsilon = 1e-6);
        assert_relative_eq!(data[2], 0.66524094, epsilon = 1e-6);
        assert_relative_eq!(data.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_inplace_negatives() {
        let mut data = vec![-1.0, 0.0, 1.0];
        softmax_inplace(&mut data);
        assert_relative_eq!(data[0], 0.09003057, epsilon = 1e-6);
        assert_relative_eq!(data[1], 0.24472847, epsilon = 1e-6);
        assert_relative_eq!(data[2], 0.66524094, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_inplace_empty() {
        let mut data: Vec<f32> = vec![];
        softmax_inplace(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_softmax_1d_inplace() {
        let mut data = arr1(&[1.0, 2.0, 3.0]);
        softmax_1d_inplace(&mut data);
        assert_relative_eq!(data[0], 0.09003057, epsilon = 1e-6);
        assert_relative_eq!(data[1], 0.24472847, epsilon = 1e-6);
        assert_relative_eq!(data[2], 0.66524094, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_1d_golden() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let golden_data = vec![0.011656, 0.031685, 0.086129, 0.234122, 0.636409];

        let mut input = Array1::from_vec(input_data);
        let golden = Array1::from_vec(golden_data);

        softmax_1d_inplace(&mut input);

        let max_diff = (&input - &golden)
            .mapv(|x| x.abs())
            .iter()
            .fold(0.0f32, |a, &b| a.max(b));
        assert!(max_diff < 1e-5, "softmax 1d mismatch: {}", max_diff);
    }

    #[test]
    fn test_softmax_4d_golden() {
        let input_data = vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
        ];
        let golden_data = vec![
            0.213838, 0.236328, 0.261183, 0.288651, 0.213838, 0.236328, 0.261183, 0.288651,
            0.213838, 0.236328, 0.261183, 0.288651, 0.213838, 0.236328, 0.261183, 0.288651,
        ];

        let mut input = Array4::from_shape_vec((1, 2, 2, 4), input_data).unwrap();
        let golden = Array4::from_shape_vec((1, 2, 2, 4), golden_data).unwrap();

        softmax_4d_inplace(&mut input);

        let max_diff = (&input - &golden)
            .mapv(|x| x.abs())
            .iter()
            .fold(0.0f32, |a, &b| a.max(b));
        assert!(max_diff < 1e-5, "softmax 4d mismatch: {}", max_diff);
    }

    #[test]
    fn test_softmax_4d_axis() {
        let mut scores = Array::from_shape_vec(
            (1, 2, 2, 3),
            vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, -1.0, 0.0, 1.0, 5.0, 5.0, 5.0],
        )
        .unwrap();

        softmax_4d_inplace(&mut scores);

        let row1 = scores.slice(s![0, 0, 0, ..]);
        assert_relative_eq!(row1.sum(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(row1[0_usize], 0.09003057, epsilon = 1e-6);

        let row2 = scores.slice(s![0, 0, 1, ..]);
        assert_relative_eq!(row2.sum(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(row2[0_usize], 0.8668133, epsilon = 1e-6);

        let row3 = scores.slice(s![0, 1, 0, ..]);
        assert_relative_eq!(row3.sum(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(row3[2_usize], 0.66524094, epsilon = 1e-6);

        let row4 = scores.slice(s![0, 1, 1, ..]);
        assert_relative_eq!(row4.sum(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(row4[0_usize], 1.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let mut input = Array4::from_shape_vec((1, 1, 1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        softmax_4d_inplace(&mut input);

        assert_relative_eq!(input.sum(), 1.0, epsilon = 1e-6);
        assert!(input[[0, 0, 0, 2]] > input[[0, 0, 0, 1]]);
        assert!(input[[0, 0, 0, 1]] > input[[0, 0, 0, 0]]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut input = Array4::from_shape_vec((1, 1, 1, 3), vec![1000.0, 1001.0, 1002.0]).unwrap();
        softmax_4d_inplace(&mut input);

        assert_relative_eq!(input.sum(), 1.0, epsilon = 1e-6);
        assert!(!input.iter().any(|x| x.is_nan()));
    }
}

#[cfg(test)]
mod quick_gelu_tests {
    use super::*;

    /// CLIP's QuickGELU is `x * sigmoid(1.702x)`, checked against PyTorch.
    #[test]
    fn quick_gelu_matches_reference() {
        // Generated with PyTorch: `x * torch.sigmoid(1.702 * x)`.
        let cases = [
            (-3.0f32, -0.0180713106),
            (-1.0, -0.1542042345),
            (-0.5, -0.1496115625),
            (0.0, 0.0),
            (0.5, 0.3503884375),
            (1.0, 0.8457957506),
            (3.0, 2.9819288254),
        ];
        for (x, want) in cases {
            let got = quick_gelu_scalar(x);
            assert!(
                (got - want).abs() < 1e-6,
                "quick_gelu({x}) = {got}, expected {want}"
            );
        }
    }

    /// The two GELU forms are close but not equal, which is why a checkpoint has
    /// to declare which one it was trained with rather than getting a default.
    #[test]
    fn quick_gelu_is_not_gelu_new() {
        let diff = (quick_gelu_scalar(1.0) - gelu_new_scalar(1.0)).abs();
        assert!(diff > 1e-3, "expected the two forms to differ, got {diff}");
    }
}
