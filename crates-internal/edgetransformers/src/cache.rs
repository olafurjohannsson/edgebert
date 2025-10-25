use std::any::Any;
use ndarray::Array3;
use std::sync::Arc;
pub use crate::traits::Cache;

/// CPU-based KV cache storing key and value tensors in ndarray format
pub struct CpuKVCache {
    /// Cached key states per layer: Vec[num_layers] of (batch, seq_len, hidden_size)
    pub keys: Vec<Option<Array3<f32>>>,
    
    /// Cached value states per layer: Vec[num_layers] of (batch, seq_len, hidden_size)
    pub values: Vec<Option<Array3<f32>>>,
    
    /// Current sequence length
    seq_length: usize,
}

impl CpuKVCache {
    /// Create a new empty cache with the specified number of layers
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![None; num_layers],
            values: vec![None; num_layers],
            seq_length: 0,
        }
    }
    
    /// Update the cache for a specific layer by appending new K, V states
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_keys: Array3<f32>,
        new_values: Array3<f32>,
    ) -> anyhow::Result<()> {
        if layer_idx >= self.keys.len() {
            anyhow::bail!("Layer index {} out of bounds", layer_idx);
        }
        
        // Concatenate with existing cache along sequence dimension
        self.keys[layer_idx] = Some(match &self.keys[layer_idx] {
            None => new_keys.clone(),
            Some(cached) => ndarray::concatenate(
                ndarray::Axis(1),
                &[cached.view(), new_keys.view()],
            )?,
        });
        
        self.values[layer_idx] = Some(match &self.values[layer_idx] {
            None => new_values.clone(),
            Some(cached) => ndarray::concatenate(
                ndarray::Axis(1),
                &[cached.view(), new_values.view()],
            )?,
        });
        
        // Update sequence length from first layer
        if layer_idx == 0 {
            self.seq_length = self.keys[0].as_ref().unwrap().shape()[1];
        }
        
        Ok(())
    }
    
    /// Get cached K, V for a specific layer (returns None if not cached yet)
    pub fn get(&self, layer_idx: usize) -> Option<(&Array3<f32>, &Array3<f32>)> {
        if layer_idx >= self.keys.len() {
            return None;
        }
        
        match (&self.keys[layer_idx], &self.values[layer_idx]) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }
}

impl Cache for CpuKVCache {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    
    fn get_seq_length(&self) -> usize {
        self.seq_length
    }
    
    fn clear(&mut self) {
        for k in &mut self.keys {
            *k = None;
        }
        for v in &mut self.values {
            *v = None;
        }
        self.seq_length = 0;
    }
}

/// GPU-based KV cache storing key and value tensors as wgpu::Buffer
pub struct GpuKVCache {
    /// Cached key buffers per layer
    pub keys: Vec<Option<Arc<wgpu::Buffer>>>,
    
    /// Cached value buffers per layer
    pub values: Vec<Option<Arc<wgpu::Buffer>>>,
    
    /// Shapes of cached tensors per layer: (batch, seq_len, hidden_size)
    pub shapes: Vec<Option<(usize, usize, usize)>>,
    
    /// Current sequence length
    seq_length: usize,
}

impl GpuKVCache {
    /// Create a new empty GPU cache with the specified number of layers
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![None; num_layers],
            values: vec![None; num_layers],
            shapes: vec![None; num_layers],
            seq_length: 0,
        }
    }
    
    /// Update the cache for a specific layer with new GPU buffers
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_key_buffer: Arc<wgpu::Buffer>,
        new_value_buffer: Arc<wgpu::Buffer>,
        shape: (usize, usize, usize),
    ) -> anyhow::Result<()> {
        if layer_idx >= self.keys.len() {
            anyhow::bail!("Layer index {} out of bounds", layer_idx);
        }
        
        // For now, just replace (concatenation on GPU requires additional shader)
        // TODO: Implement GPU buffer concatenation
        self.keys[layer_idx] = Some(new_key_buffer);
        self.values[layer_idx] = Some(new_value_buffer);
        self.shapes[layer_idx] = Some(shape);
        
        // Update sequence length
        if layer_idx == 0 {
            self.seq_length = shape.1;
        }
        
        Ok(())
    }
    
    /// Get cached K, V buffers for a specific layer
// In cache.rs, line 160:
pub fn get(&self, layer_idx: usize) -> Option<(&Arc<wgpu::Buffer>, &Arc<wgpu::Buffer>, (usize, usize, usize))> {
    if layer_idx >= self.keys.len() {
        return None;
    }
    
    match (&self.keys[layer_idx], &self.values[layer_idx], &self.shapes[layer_idx]) {
        (Some(k), Some(v), Some(shape)) => Some((k, v, *shape)),  // Add * here!
        _ => None,
    }
}
}

impl Cache for GpuKVCache {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    
    fn get_seq_length(&self) -> usize {
        self.seq_length
    }
    
    fn clear(&mut self) {
        for k in &mut self.keys {
            *k = None;
        }
        for v in &mut self.values {
            *v = None;
        }
        for s in &mut self.shapes {
            *s = None;
        }
        self.seq_length = 0;
    }
}