//! Layer normalization implementation

use ndarray::{Array1, Array3, Axis};

/// Layer normalization
pub struct LayerNorm {
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Apply layer norm to a 3D tensor
pub fn forward_3d(&self, hidden: &Array3<f32>) -> Array3<f32> {
    // 1. Calculate the mean and variance along the last axis (the feature dimension).
    //    `keep_dims` is false, so these will be 2D arrays.
    let mean = hidden.mean_axis(Axis(2)).unwrap();
    let variance = hidden.var_axis(Axis(2), 0.0);

    // 2. Expand the dimensions of the mean and variance so they can be broadcast
    //    for subtraction and division. Shape goes from [batch, seq] to [batch, seq, 1].
    let mean_expanded = mean.insert_axis(Axis(2));
    let var_expanded = variance.insert_axis(Axis(2));

    // 3. Normalize the hidden state: (x - mean) / sqrt(var + epsilon)
    let inv_std = (&var_expanded + self.eps).mapv(|x| 1.0 / x.sqrt());
    let normalized_hidden = (hidden - &mean_expanded) * &inv_std;
    
    // 4. Apply the learnable parameters: y = normalized * weight + bias
    //    This correctly broadcasts the 1D weight and bias vectors.
    let scaled = normalized_hidden * &self.weight;
    let shifted = scaled + &self.bias;

    shifted
}
}
