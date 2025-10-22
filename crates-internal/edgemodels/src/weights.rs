//! Weight loading utilities for GPT models

use anyhow::Result;
use ndarray::{Array1, Array2};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

pub struct ModelWeights {
    tensors: HashMap<String, Vec<f32>>,
    shapes: HashMap<String, Vec<usize>>,
    pub config_json: String,
}

impl ModelWeights {
    pub fn load(path: &Path) -> Result<Self> {
        let weights_file = path.join("model.safetensors");
        let data = std::fs::read(weights_file)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let mut tensor_data = HashMap::new();
        let mut shapes = HashMap::new();

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let dtype = view.dtype();
            // let data_bytes = view.data();

            let data: Vec<f32> = match dtype {
                // Dtype::F32 => {
                //     let (prefix, floats, suffix) = unsafe { data_bytes.align_to::<f32>() };
                //     // These asserts ensure that the memory alignment is perfect, which it should be for safetensors.
                //     assert!(prefix.is_empty(), "Data was not aligned for F32");
                //     assert!(suffix.is_empty(), "Data was not aligned for F32");
                //     floats.to_vec()
                // }
                Dtype::F32 => view
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                Dtype::F16 => {
                    use half::f16;
                    view.data()
                        .chunks_exact(2)
                        .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                        .collect()
                }
                Dtype::BF16 => {
                    use half::bf16;
                    view.data()
                        .chunks_exact(2)
                        .map(|b| bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                        .collect()
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported tensor dtype {:?} for tensor '{}'",
                        dtype,
                        name
                    ));
                }
            };

            shapes.insert(name.to_string(), shape);
            tensor_data.insert(name.to_string(), data);
        }

        let config_file = path.join("config.json");
        let config_json = std::fs::read_to_string(config_file)?;

        Ok(Self {
            tensors: tensor_data,
            shapes,
            config_json,
        })
    }
    //     nitial Symptom: The final output of the 12-layer encoder diverged significantly from the Python reference, even when the first layer's output matched.
    // ompounding error. Here’s what it means step-by-step:
    // Layer 0: We first checked the output of just the first encoder layer. The Rust output (e.g., 0.12344) was almost identical to the Python output (e.g., 0.12345). This tiny difference is normal floating-point error and is not a bug. We thought we were on the right track.
    // Layer 1: The second layer in Rust doesn't start with the "perfect" Python value (0.12345); it starts with its own slightly different result (0.12344). It then performs its own calculations, which also have a tiny error. The output of Layer 1 is now a bit more different.
    // Layer 2 to Layer 11: This process repeats. At each of the 12 layers, the small error from the previous layer is fed in, and the current layer adds its own tiny error on top.
    // Final Output: After this "snowball effect" rolls through all 12 layers, the tiny initial difference has grown into a large, significant one. The final Rust output (0.987...) looks completely different from the final Python output (1.234...).
    // The Core Bug (Activations): The Rust code used a mathematically pure erf-based GELU function, but the model was trained with a slightly different tanh approximation. This tiny initial difference compounded over 12 layers, causing the final divergence.
    // Tensor Shape & Reshaping Bugs:
    // The decoder's causal mask had a critical shape bug. During generation with a KV cache, the input sequence length is 1, so the attention mask must be shaped [1, total_sequence_length] to allow the new token to see all previous tokens.
    // Broadcasting in LayerNorm required explicit dimension expansion. The 2D mean and variance tensors had to be reshaped to [batch, seq, 1] to correctly normalize the 3D hidden state.
    // Scaling Issues & Order of Operations:
    // The most elusive bug was the attention scaling order. The PyTorch implementation scales the 3D Query tensor before reshaping and matrix multiplication, not the final 4D scores matrix after. Replicating this exact order was essential.
    // The scale_factor itself (1 / sqrt(head_dim)) was a red herring; it was always correct, but its point of application in the pipeline was wrong.
    // The "Invisible" Bug (Memory Layout):
    // A major source of confusion was the memory layout (strides) of ndarray arrays. A tensor modified with .permuted_axes() is a non-contiguous "view." Matrix multiplication functions can misinterpret these strides, producing garbage output even with mathematically correct inputs.
    // The solution was to force tensors into a standard memory layout with .as_standard_layout().to_owned() immediately before the matmul call.
    // Classic Generation Bugs:
    // The model initially produced repetitive loops, which was fixed by correctly applying the repetition_penalty to the logits at each step.
    // The model was confused at the start of generation, which was solved by using the correct BOS (Beginning of Sentence) token to initialize the decoder, not EOS.
    #[cfg(target_arch = "wasm32")]
    pub fn from_bytes(data: &[u8], config_json: &str) -> Result<Self> {
        let tensors = SafeTensors::deserialize(data)?;

        let mut tensor_data = HashMap::new();
        let mut shapes = HashMap::new();

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let data: Vec<f32> = view
                .data()
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            shapes.insert(name.to_string(), shape);
            tensor_data.insert(name.to_string(), data);
        }

        Ok(Self {
            tensors: tensor_data,
            shapes,
            config_json: config_json.to_string(),
        })
    }

    pub fn get_array1(&self, name: &str) -> Result<Array1<f32>> {
        let data = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;
        let shape = &self.shapes[name];
        anyhow::ensure!(
            shape.len() == 1,
            "Expected 1D tensor, got shape {:?}",
            shape
        );
        Ok(Array1::from_vec(data.clone()))
    }

    pub fn get_array2(&self, name: &str) -> Result<Array2<f32>> {
        let data = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;
        let shape = &self.shapes[name];
        Array2::from_shape_vec((shape[0], shape[1]), data.clone())
            .map_err(|e| anyhow::anyhow!("Shape error: {}", e))
    }
}
