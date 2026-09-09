//! CPU-based linear layer with multi-dtype support.

use crate::cpu::{
    kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0},
    ops,
};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::tensor::raw_tensor::TensorView;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::WgpuContext;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::gpu::GpuTensor;
use crate::linear_layer::LinearLayerBuilder;
use crate::tensor::{DType, QuantizedMatrix};
use crate::utils::tensor_ops;
use crate::weights::ModelWeights;
use anyhow::{Result, anyhow};
use half::bf16;
use ndarray::{Array1, Array2, ArrayView2};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use std::borrow::Cow;
use std::sync::Arc;

/// Strategy for F32 matrix multiplication dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32MatmulStrategy {
    Faer,
    CustomSimd,
    FaerOutIn,
}

/// A CPU-based linear transformation layer (y = xW^T + b).
#[derive(Clone, Debug)]
pub struct LinearLayer {
    /// The weight matrix in one of several supported formats.
    pub data: LinearData,

    /// Optional bias vector, always stored as F32.
    pub bias: Option<Array1<f32>>,

    /// Strategy for F32 matmul dispatch (Faer vs CustomSimd).
    pub f32_strategy: F32MatmulStrategy,
}

/// Weight data storage for a linear layer.
#[allow(non_camel_case_types)]
#[derive(Clone, Debug)]
pub enum LinearData {
    /// 32-bit IEEE 754 floating point weights.
    F32(Arc<Array2<f32>>),

    /// 16-bit brain floating point weights.
    BF16(Arc<Array2<bf16>>),

    /// 16-bit IEEE half-precision floating point weights.
    F16(Arc<Array2<half::f16>>),

    /// 8-bit block-quantized weights (32 elements per block).
    Q8_0(Arc<QuantizedMatrix<BlockQ8_0>>),

    /// 4-bit block-quantized weights with K-quants (256 elements per block).
    Q4_K(Arc<QuantizedMatrix<BlockQ4_K>>),

    /// 6-bit block-quantized weights with K-quants.
    Q6_K(Arc<QuantizedMatrix<BlockQ6_K>>),
}
impl LinearData {
    /// Returns the data type of the stored weights.
    pub fn dtype(&self) -> DType {
        match self {
            LinearData::F32(_) => DType::F32,
            LinearData::BF16(_) => DType::BF16,
            LinearData::F16(_) => DType::F16,
            LinearData::Q8_0(_) => DType::Q8_0,
            LinearData::Q6_K(_) => DType::Q6_K,
            LinearData::Q4_K(_) => DType::Q4_K,
        }
    }
}

impl LinearLayer {
    /// Threshold for switching from vec kernel to batched 4x3 kernel.
    /// Based on benchmarks: vec kernel wins for m < ~1000, batched wins for m >= ~1000.
    #[allow(dead_code, reason = "kept while the new dispatch is being measured")]
    const BATCH_KERNEL_THRESHOLD: usize = 1000;

    /// Above this many rows faer beats the hand-written kernels.
    ///
    /// Re-swept after the mid band moved to column-parallel 4x3 tiles. Against
    /// the old row-parallel vector kernel every candidate landed within 2.4% and
    /// 256 was as good as any; with the tile kernel, 512 is better in 19 of 60
    /// rows and worse in 2, because the tile stays ahead of faer roughly twice as
    /// far up. 1024 is a wash, 13 better and 11 worse.
    const FAER_THRESHOLD: usize = 512;

    /// Creates a new zero-initialized `LinearLayer` with the specified dimensions and dtype.
    pub fn new(out_features: usize, in_features: usize, dtype: DType) -> Self {
        match dtype {
            DType::F32 => Self::new_f32(Array2::<f32>::zeros((out_features, in_features)), None),
            DType::BF16 => Self::new_bf16(Array2::<bf16>::zeros((out_features, in_features)), None),
            _ => {
                panic!("Unsupported dtype for LinearLayer::new: {:?}", dtype);
            }
        }
    }

    /// Returns a builder for constructing a `LinearLayer` from model weights.
    pub fn builder<'a>(weights: &'a ModelWeights, name: &'a str) -> LinearLayerBuilder<'a> {
        LinearLayerBuilder::new(weights, name)
    }
    /// Creates a new F32 `LinearLayer` from a weight matrix and an optional bias.
    pub fn new_f32(weights: Array2<f32>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::F32(Arc::new(weights)),
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    pub fn from_arc_f32(weights: Arc<Array2<f32>>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::F32(weights), // No copy!
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    /// Creates a new BF16 `LinearLayer` from a weight matrix and an optional bias.
    pub fn new_bf16(weights: Array2<bf16>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::BF16(Arc::new(weights)),
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    pub fn from_arc_bf16(weights: Arc<Array2<bf16>>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::BF16(weights), // No copy!
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    // Q8_0 Sharing
    pub fn from_arc_q8_0(
        weights: Arc<QuantizedMatrix<BlockQ8_0>>,
        bias: impl Into<Option<Array1<f32>>>,
    ) -> Self {
        Self {
            data: LinearData::Q8_0(weights),
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    pub fn new_f32_with_strategy(
        weights: Array2<f32>,
        bias: impl Into<Option<Array1<f32>>>,
        strategy: F32MatmulStrategy,
    ) -> Self {
        Self {
            data: LinearData::F32(Arc::new(weights)),
            bias: bias.into(),
            f32_strategy: strategy,
        }
    }

    /// Computes matrix multiplication writing to a pre-allocated output buffer.
    #[inline]
    pub fn matmul_noalloc(&self, input: &ArrayView2<f32>, output: &mut Array2<f32>) {
        match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::CustomSimd => {
                    let (m, _) = input.dim();
                    let bias = self.bias.as_ref().map(|b| b.as_slice().unwrap());

                    // Measured on MiniLM and Qwen 0.5B shapes across m from 1 to
                    // 2048: the blocked kernel wins at a single row, the vector
                    // kernel from there to a few hundred, and faer above that by
                    // 1.3x to 1.6x. One threshold could not express that.
                    if m == 1 {
                        ops::matmul::matmul_2d_f32_batched_noalloc(input, &w.view(), bias, output);
                    } else if m < Self::FAER_THRESHOLD {
                        // Column-parallel bands worked in 4x3 register tiles.
                        //
                        // Splitting by column rather than by row was worth 26 of
                        // 60 rows on its own, because at m below the thread count
                        // row-splitting starves threads and drags the whole
                        // weight matrix through cache once per row: a short
                        // MiniLM document went from 5.5ms to 2.8ms.
                        //
                        // The 4x3 tile then replaced one output at a time, 12
                        // FMAs per 7 loads against one per two, and took a 220
                        // token document on bge-m3 from 560ms to 336ms and on
                        // distilbert from 69ms to 45ms. Neither change lost a row
                        // that was not on the faer path.
                        ops::matmul::matmul_2d_f32_tile43_par_n(input, &w.view(), bias, output);
                    } else {
                        ops::matmul::matmul_2d_f32_faer_noalloc(input, &w.view(), bias, output);
                    }
                }
                F32MatmulStrategy::Faer | F32MatmulStrategy::FaerOutIn => {
                    // Faer doesn't have a no-alloc API, fall back to allocating
                    // and copy to output
                    let result = self.matmul(input);
                    output.assign(&result);
                }
            },
            LinearData::BF16(_)
            | LinearData::F16(_)
            | LinearData::Q8_0(_)
            | LinearData::Q6_K(_)
            | LinearData::Q4_K(_) => {
                // Quantized/BF16 paths don't have no-alloc versions yet
                // Fall back to allocating and copy
                let result = self.matmul(input);
                output.assign(&result);
            }
        }
    }

    /// Computes matrix multiplication with automatic kernel dispatch.
    #[inline]
    pub fn matmul(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::CustomSimd => {
                    let (m, _) = input.dim();
                    if m == 1 {
                        // Decode path - add bias manually after
                        let mut result = ops::matmul::matmul_2d_cpu_f32(input, &w.view());
                        if let Some(b) = &self.bias {
                            // Single row, just add directly
                            let out = result.as_slice_mut().unwrap();
                            let bias = b.as_slice().unwrap();
                            for (o, &b) in out.iter_mut().zip(bias.iter()) {
                                *o += b;
                            }
                        }
                        result
                    } else {
                        // Batch path - pass bias to kernel (fused, faster)
                        ops::matmul::matmul_2d_cpu_f32_batched(
                            input,
                            &w.view(),
                            self.bias.as_ref().map(|b| b.as_slice().unwrap()),
                        )
                    }
                }
                F32MatmulStrategy::Faer => {
                    let mut result = tensor_ops::matmul_2d_faer(input, &w.view());
                    if let Some(b) = &self.bias {
                        for mut row in result.rows_mut() {
                            row += b;
                        }
                    }
                    result
                }
                F32MatmulStrategy::FaerOutIn => {
                    let mut result = ops::matmul::matmul_2d_cpu_f32_faer(input, &w.view());
                    if let Some(b) = &self.bias {
                        for mut row in result.rows_mut() {
                            row += b;
                        }
                    }
                    result
                }
            },
            LinearData::BF16(w) => {
                let mut result = ops::matmul::matmul_2d_cpu_bf16(input, &w.view());
                if let Some(b) = &self.bias {
                    for mut row in result.rows_mut() {
                        row += b;
                    }
                }
                result
            }
            LinearData::F16(_w) => {
                unimplemented!("F16 LinearLayer matmul not implemented yet");
            }
            LinearData::Q8_0(w) => {
                let mut result = ops::matmul::matmul_2d_cpu_q8_0(input, &w.blocks);
                if let Some(b) = &self.bias {
                    for mut row in result.rows_mut() {
                        row += b;
                    }
                }
                result
            }
            LinearData::Q6_K(w) => {
                let mut result = ops::matmul::matmul_2d_cpu_q6_k(input, &w.blocks);
                if let Some(b) = &self.bias {
                    for mut row in result.rows_mut() {
                        row += b;
                    }
                }
                result
            }
            LinearData::Q4_K(w) => {
                let mut result = ops::matmul::matmul_2d_cpu_q4_k(input, &w.blocks);
                if let Some(b) = &self.bias {
                    for mut row in result.rows_mut() {
                        row += b;
                    }
                }
                result
            }
        }
    }

    /// Get reference to bias if present
    pub fn bias(&self) -> Option<&Array1<f32>> {
        self.bias.as_ref()
    }

    /// Get a view of the weights [out_features, in_features]
    pub fn weights_view(&self) -> ArrayView2<'_, f32> {
        match self.data {
            crate::linear_layer::LinearData::F32(ref w) => w.view(),
            _ => panic!("Only f32 LinearLayer supported in optimized path"),
        }
    }

    /// Get weights as contiguous slice [out_features * in_features]
    pub fn weights_slice(&self) -> &[f32] {
        match self.data {
            crate::linear_layer::LinearData::F32(ref w) => w
                .as_slice()
                .expect("LinearLayer weights must be contiguous for fused ops"),
            _ => panic!("Only f32 LinearLayer supported in optimized path"),
        }
    }

    pub fn weights_slice_bf16(&self) -> Option<&[bf16]> {
        match &self.data {
            LinearData::BF16(w) => w.as_slice(),
            _ => None,
        }
    }

    /// Get bias as slice if present
    pub fn bias_slice(&self) -> Option<&[f32]> {
        self.bias.as_ref().map(|b| b.as_slice().unwrap())
    }

    /// Converts this layer to a quantized format.
    pub fn to_quantized(&self, dtype: DType) -> Result<Self> {
        if self.bias.is_some() {
            log::warn!("Quantizing a layer that has a bias. Bias will be discarded.");
        }

        let quantized_data = match (&self.data, dtype) {
            (LinearData::BF16(w), DType::Q8_0) => {
                let w_f32 = w.mapv(|v| v.to_f32());
                let blocks = crate::cpu::kernels::quantize::quantize_matrix_q8_0(&w_f32)?;
                let shape = [w.shape()[0], w.shape()[1]];
                LinearData::Q8_0(Arc::new(QuantizedMatrix { blocks, shape }))
            }
            (LinearData::F32(w), DType::Q8_0) => {
                let blocks = crate::cpu::kernels::quantize::quantize_matrix_q8_0(w)?;
                let shape = [w.shape()[0], w.shape()[1]];
                LinearData::Q8_0(Arc::new(QuantizedMatrix { blocks, shape }))
            }
            (d, t) => {
                return Err(anyhow!(
                    "Unsupported quantization from {:?} to {:?}",
                    d.dtype(),
                    t
                ));
            }
        };

        Ok(Self {
            data: quantized_data,
            bias: None, // Bias is not used in quantized matmul for simplicity
            f32_strategy: self.f32_strategy,
        })
    }

    /// Returns the shape of the weight matrix as `[out_features, in_features]`.
    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        [self.out_features(), self.in_features()]
    }

    /// Returns the number of output features.
    #[must_use]
    pub fn out_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => w.shape()[1],
                F32MatmulStrategy::CustomSimd => w.shape()[0],
                F32MatmulStrategy::FaerOutIn => w.shape()[0],
            },
            LinearData::BF16(w) => w.shape()[0],
            LinearData::F16(w) => w.shape()[0],
            LinearData::Q8_0(w) => w.shape[0],
            LinearData::Q4_K(w) => w.shape[0],
            LinearData::Q6_K(w) => w.shape[0],
        }
    }

    /// Returns the number of input features.
    #[must_use]
    pub fn in_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => w.shape()[0],
                F32MatmulStrategy::CustomSimd => w.shape()[1],
                F32MatmulStrategy::FaerOutIn => w.shape()[1],
            },
            LinearData::BF16(w) => w.shape()[1],
            LinearData::F16(w) => w.shape()[1],
            LinearData::Q8_0(w) => w.shape[1],
            LinearData::Q4_K(w) => w.shape[1],
            LinearData::Q6_K(w) => w.shape[1],
        }
    }

    /// Returns the data type of the weight matrix.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Returns `true` if this layer has a bias term.
    #[must_use]
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    /// Uploads the bias tensor (if present) to the GPU.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn bias_to_gpu(
        &self,
        ctx: &Arc<crate::WgpuContext>,
    ) -> Result<Option<crate::gpu::GpuTensor>> {
        if let Some(bias) = &self.bias {
            let gpu_bias = crate::gpu::GpuTensor::from_ndarray(ctx, bias)?;
            Ok(Some(gpu_bias))
        } else {
            Ok(None)
        }
    }

    /// Uploads the weight matrix to the GPU.
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn to_gpu(&self, ctx: &Arc<WgpuContext>) -> Result<GpuTensor> {
        self.to_gpu_tensor(ctx, "to_gpu")
    }

    /// Converts the linear layer's weight matrix to a `GpuTensor`.
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn to_gpu_tensor(&self, ctx: &Arc<WgpuContext>, label: &str) -> Result<GpuTensor> {
        match &self.data {
            LinearData::F32(arr) => {
                // For F32, we can get a direct slice of the data.
                let layout = arr.as_standard_layout();

                let slice = layout
                    .as_slice()
                    .ok_or_else(|| anyhow!("F32 tensor is not contiguous"))?;

                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(slice)),
                    shape: arr.shape().to_vec(),
                    dtype: DType::F32,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
            LinearData::BF16(arr) => {
                let layout = arr.as_standard_layout();

                let slice = layout
                    .as_slice()
                    .ok_or_else(|| anyhow!("BF16 tensor is not contiguous"))?;

                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(slice)),
                    shape: arr.shape().to_vec(),
                    dtype: DType::BF16,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
            LinearData::F16(arr) => {
                let layout = arr.as_standard_layout();

                let slice = layout
                    .as_slice()
                    .ok_or_else(|| anyhow!("F16 tensor is not contiguous"))?;

                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(slice)),
                    shape: arr.shape().to_vec(),
                    dtype: DType::F16,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
            LinearData::Q8_0(matrix) => {
                // Repacked to 9 words per block, the layout the Q8_0 kernels index.
                // See `repack_q8_0`; shipping the raw 34-byte blocks puts `qs` out of
                // word alignment and the shader reads garbage.
                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Owned(crate::gpu::tensor::repack_q8_0(&matrix.blocks)),
                    shape: matrix.shape.to_vec(),
                    dtype: DType::Q8_0,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
            LinearData::Q4_K(matrix) => {
                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(&matrix.blocks)),
                    shape: matrix.shape.to_vec(),
                    dtype: DType::Q4_K,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
            LinearData::Q6_K(matrix) => {
                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(&matrix.blocks)),
                    shape: matrix.shape.to_vec(),
                    dtype: DType::Q6_K,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
        }
    }
}

/// Converts an F32 weight matrix into a `LinearLayer` with no bias.
impl From<Array2<f32>> for LinearLayer {
    fn from(weights: Array2<f32>) -> Self {
        LinearLayer::new_f32(weights, None)
    }
}

/// Converts an F32 weight matrix and bias into a `LinearLayer`.
impl From<(Array2<f32>, Array1<f32>)> for LinearLayer {
    fn from((weights, bias): (Array2<f32>, Array1<f32>)) -> Self {
        LinearLayer::new_f32(weights, Some(bias))
    }
}

/// Converts a BF16 weight matrix into a `LinearLayer` with no bias.
impl From<Array2<bf16>> for LinearLayer {
    fn from(weights: Array2<bf16>) -> Self {
        LinearLayer::new_bf16(weights, None)
    }
}

/// Converts a BF16 weight matrix and bias into a `LinearLayer`.
impl From<(Array2<bf16>, Array1<f32>)> for LinearLayer {
    fn from((weights, bias): (Array2<bf16>, Array1<f32>)) -> Self {
        LinearLayer::new_bf16(weights, Some(bias))
    }
}
