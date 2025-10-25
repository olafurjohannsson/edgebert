//! Generic, thread-safe utilities for loading model weights from `.safetensors` files.
//!
//! This module provides a `ModelWeights` struct that can deserialize tensors
//! from various formats (F32, F16) into a common `f32` representation in memory.
//! It is designed to be model-agnostic, providing a consistent API for accessing
//! weight tensors by name, which is then used by the generic model stacks.

use anyhow::{Context, Result, anyhow};
use ndarray::{Array1, Array2};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// A generic container for model weights loaded from a `.safetensors` file.
///
/// This struct holds all tensors in a hash map, converted to `f32` for use with
/// `ndarray`. It also stores the model's `config.json` content, making it a
/// self-contained source of truth for constructing a model.
pub struct ModelWeights {
    /// A map from tensor names to their data (as `Vec<f32>`) and shape.
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
    /// The raw JSON string content of the model's `config.json` file.
    pub config_json: String,
}

impl ModelWeights {
    /// Creates a new `ModelWeights` instance by loading from a directory.
    ///
    /// It expects to find `model.safetensors` and `config.json` in the specified path.
    pub fn new(path: &Path) -> Result<Self> {
        let weights_file = path.join("model.safetensors");
        let data = fs::read(&weights_file)
            .with_context(|| format!("Failed to read weights file at: {:?}", weights_file))?;

        let config_file = path.join("config.json");
        let config_json = fs::read_to_string(&config_file)
            .with_context(|| format!("Failed to read config file at: {:?}", config_file))?;

        Self::from_bytes(&data, &config_json)
    }

    /// Creates a new `ModelWeights` instance from in-memory byte slices.
    ///
    /// This is useful for environments like WASM where the file system is not
    /// directly accessible. It handles deserialization and data type conversion.
    pub fn from_bytes(data: &[u8], config_json: &str) -> Result<Self> {
        let tensors_view = SafeTensors::deserialize(data)?;
        let mut tensors = HashMap::new();

        for (name, view) in tensors_view.tensors() {
            let shape = view.shape().to_vec();
            let tensor_data_opt: Option<Vec<f32>> = match view.dtype() {
                Dtype::F32 => Some(
                    view.data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                ),
                Dtype::F16 => {
                    use half::f16;
                    Some(
                        view.data()
                            .chunks_exact(2)
                            .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                            .collect(),
                    )
                }
                dtype => {
                    // todo: log/tracing 
                    println!(
                        "   [Weight Loader Warning] Skipping tensor '{}' with unsupported dtype: {:?}",
                        name, dtype
                    );
                    None
                }
            };

            if let Some(tensor_data) = tensor_data_opt {
                tensors.insert(name.to_string(), (tensor_data, shape));
            }
        }

        Ok(Self {
            tensors,
            config_json: config_json.to_string(),
        })
    }

    /// Retrieves a 1D tensor (`Array1<f32>`) by name.
    pub fn get_array1(&self, name: &str) -> Result<Array1<f32>> {
        let (data, shape) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' 1d not found in weights", name))?;
        anyhow::ensure!(
            shape.len() == 1,
            "Expected 1D tensor for '{}', but got shape {:?}",
            name,
            shape
        );
        Ok(Array1::from_vec(data.clone()))
    }

    /// Retrieves a 2D tensor (`Array2<f32>`) by name.
    pub fn get_array2(&self, name: &str) -> Result<Array2<f32>> {
        let (data, shape) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' 2d not found in weights", name))?;
        anyhow::ensure!(
            shape.len() == 2,
            "Expected 2D tensor for '{}', but got shape {:?}",
            name,
            shape
        );
        Array2::from_shape_vec((shape[0], shape[1]), data.clone())
            .with_context(|| format!("Shape error for tensor '{}'", name))
    }
}
