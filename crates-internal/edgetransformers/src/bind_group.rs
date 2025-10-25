use crate::wgpu_context::WgpuContext;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, Buffer, CommandEncoder, ComputePipeline, Device, Queue, include_wgsl};

pub struct BindGroupCache {
    uniform_pools: HashMap<u64, Vec<Arc<Buffer>>>,
    pool_indices: HashMap<u64, usize>,
    pool_size: usize,
}

impl BindGroupCache {
    pub fn new() -> Self {
        Self {
            uniform_pools: HashMap::new(),
            pool_indices: HashMap::new(),
            pool_size: 16,
        }
    }

    pub fn with_capacity(_capacity: usize, pool_size: usize) -> Self {
        Self {
            uniform_pools: HashMap::new(),
            pool_indices: HashMap::new(),
            pool_size,
        }
    }

    pub fn get_uniform_buffer(&mut self, device: &Device, size: u64) -> Arc<Buffer> {
        let pool = self.uniform_pools.entry(size).or_insert_with(Vec::new);
        let index = self.pool_indices.entry(size).or_insert(0);

        if pool.len() < self.pool_size {
            let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Uniform Pool"),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            pool.push(buffer.clone());
            return buffer;
        }

        let buffer = pool[*index % self.pool_size].clone();
        *index += 1;
        buffer
    }

    pub fn prepare<T: bytemuck::Pod>(
        &mut self,
        device: &Device,
        queue: &Queue,
        pipeline: &ComputePipeline,
        uniforms: &T,
        storage_buffers: &[&Buffer],
    ) -> wgpu::BindGroup {
        let uniform_size = std::mem::size_of::<T>() as u64;
        let uniform_buffer = self.get_uniform_buffer(device, uniform_size);
        
        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[*uniforms]));

        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }];

        for (i, buf) in storage_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: buf.as_entire_binding(),
            });
        }

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// Clear all pools (useful for cleanup)
    pub fn clear(&mut self) {
        self.uniform_pools.clear();
        self.pool_indices.clear();
    }

    /// Get statistics about cache usage
    pub fn stats(&self) -> CacheStats {
        let total_uniform_buffers: usize = self.uniform_pools.values().map(|p| p.len()).sum();

        CacheStats {
            uniform_pools: self.uniform_pools.len(),
            total_uniform_buffers,
        }
    }
}

#[derive(Debug)]
pub struct CacheStats {
    pub uniform_pools: usize,
    pub total_uniform_buffers: usize,
}

impl Default for BindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}