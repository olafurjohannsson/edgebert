//! Weight loading utilities for BERT models

use anyhow::Result;
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

pub struct BertModelWeights {
    tensors: HashMap<String, Vec<f32>>,
    shapes: HashMap<String, Vec<usize>>,
    pub config_json: String,
}

impl BertModelWeights {
    pub fn load(path: &Path) -> Result<Self> {
        let weights_file = path.join("model.safetensors");
        let data = std::fs::read(weights_file)?;
        let tensors = SafeTensors::deserialize(&data)?;
        
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
        
        // Load config
        let config_file = path.join("config.json");
        let config_json = std::fs::read_to_string(config_file)?;
        
        Ok(Self {
            tensors: tensor_data,
            shapes,
            config_json,
        })
    }
    
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
    
    pub fn get_scalar(&self, name: &str) -> Result<f32> {
        let data = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;
        
        anyhow::ensure!(
            data.len() == 1,
            "Expected scalar, got {} elements",
            data.len()
        );
        Ok(data[0])
    }
    
    /// Check if this is a cross-encoder model
    pub fn is_cross_encoder(&self) -> bool {
        self.tensors.contains_key("classifier.weight")
            || self.tensors.contains_key("classifier.dense.weight")
    }
}