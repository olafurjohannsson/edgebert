use ndarray::{Array2, Ix2};

// This enum holds either a standard f32 tensor or a quantized i8 tensor with its scale
pub enum QuantizableTensor {
    Float(Array2<f32>),
    Quantized(Array2<i8>, f32),
}

impl QuantizableTensor {
    pub fn dequantize(&self) -> std::borrow::Cow<Array2<f32>> {
        match self {
            QuantizableTensor::Float(arr) => std::borrow::Cow::Borrowed(arr),
            QuantizableTensor::Quantized(arr, scale) => {
                // Perform the dequantization: int8 -> float32
                let dequantized = arr.mapv(|val| val as f32 * *scale);
                std::borrow::Cow::Owned(dequantized)
            }
        }
    }
}