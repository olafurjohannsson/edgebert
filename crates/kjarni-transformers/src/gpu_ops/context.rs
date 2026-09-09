use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use tokio::sync::Mutex;
use wgpu::{
    Adapter, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits, PowerPreference,
    RequestAdapterOptions,
};

use crate::gpu::GpuTensorPool;
use crate::gpu_ops::profiler::GpuProfiler;
use crate::gpu_ops::uniforms::GpuUniformBuffer;

#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub max_buffer_size: u64,
    pub max_texture_dimension_2d: u32,
    pub max_storage_buffer_binding_size: u32,
    pub available_memory: Option<u64>,
    pub reserved_for_kv_cache: u64,
}

impl GpuMemoryInfo {
    pub fn print_summary(&self) {
        log::debug!(
            "gpu memory: max_buffer={:.2}GB, max_binding={:.2}GB, kv_reserved={:.2}GB",
            self.max_buffer_size as f64 / 1_073_741_824.0,
            self.max_storage_buffer_binding_size as f64 / 1_073_741_824.0,
            self.reserved_for_kv_cache as f64 / 1_073_741_824.0
        );
        if let Some(available) = self.available_memory {
            log::debug!(
                "available vram: {:.2}GB",
                available as f64 / 1_073_741_824.0
            );
        }
    }
}

pub struct WgpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: Adapter,
    pub memory_info: GpuMemoryInfo,
    allocated_memory: AtomicUsize,
    inference_pool: OnceLock<Arc<Mutex<GpuTensorPool>>>,
    pub profiler: GpuProfiler,
    pub uniform_arena: GpuUniformBuffer,
}

/// The largest workgroup any shader in the crate declares, `(32, 32, 1)`.
///
/// A device request that asks for less than this succeeds, and the oversized
/// pipelines then fail to build without stopping the dispatch, so the output is
/// silently NaN. Checked up front instead.
const REQUIRED_INVOCATIONS: u32 = 1024;

impl WgpuContext {
    pub async fn new() -> Result<Arc<Self>> {
        Self::with_config(GpuConfig::default()).await
    }

    pub fn is_available() -> bool {
        true
    }

    pub async fn with_config(config: GpuConfig) -> Result<Arc<Self>> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });

        // Requested, never required. A browser exposes timestamp queries only
        // behind a developer flag, and demanding them makes `request_device` fail
        // outright rather than merely losing the profiler.
        const WANTED: Features =
            Features::TIMESTAMP_QUERY.union(Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let _adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let required_features = WANTED & adapter.features();
        let timestamps = required_features.contains(Features::TIMESTAMP_QUERY);
        if !timestamps {
            log::info!("adapter has no timestamp queries; GPU profiling is disabled");
        }

        log::info!(
            "Adapter reported limits: max_storage_buffer_binding_size={} ({:.2}MB), max_buffer_size={} ({:.2}GB)",
            adapter_limits.max_storage_buffer_binding_size,
            adapter_limits.max_storage_buffer_binding_size as f64 / 1_048_576.0,
            adapter_limits.max_buffer_size,
            adapter_limits.max_buffer_size as f64 / 1_073_741_824.0,
        );

        // Five shader entry points declare more than 256 invocations per workgroup:
        // two at (32, 32, 1), two at (512, 1, 1) and one at (8, 8, 8). WebGPU only
        // guarantees 256, and an adapter at that floor does not fail loudly. It
        // produces NaN, which is far worse than refusing, so refuse.
        if adapter_limits.max_compute_invocations_per_workgroup < REQUIRED_INVOCATIONS {
            return Err(anyhow!(
                "adapter grants {} compute invocations per workgroup; kjarni needs {}. \
                 Its shaders cannot run here, so use the CPU backend instead.",
                adapter_limits.max_compute_invocations_per_workgroup,
                REQUIRED_INVOCATIONS,
            ));
        }

        let memory_info = Self::calculate_memory_info(&adapter, &config)?;
        memory_info.print_summary();

        if let Some(available) = memory_info.available_memory {
            let required_min = config.kv_cache_memory_mb * 1_048_576;
            if available < required_min {
                log::warn!(
                    "available vram ({:.2}GB) less than kv cache reservation ({:.2}GB)",
                    available as f64 / 1_073_741_824.0,
                    required_min as f64 / 1_073_741_824.0
                );
            }
        }

        let required_limits = Limits {
            max_buffer_size: memory_info.max_buffer_size,
            max_storage_buffer_binding_size: memory_info.max_storage_buffer_binding_size,
            max_texture_dimension_2d: memory_info.max_texture_dimension_2d,
            max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x.min(1024),
            max_compute_workgroup_size_y: adapter_limits.max_compute_workgroup_size_y.min(1024),
            max_compute_invocations_per_workgroup: adapter_limits
                .max_compute_invocations_per_workgroup
                .min(1024),
            ..adapter_limits.clone()
        };

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("transformer_gpu"),
                required_features,
                required_limits,
                ..Default::default()
            })
            .await?;

        log::info!(
            "device initialized: max_buffer={:.2}GB, max_binding={:.2}GB",
            memory_info.max_buffer_size as f32 / 1_073_741_824.0,
            memory_info.max_storage_buffer_binding_size as f32 / 1_073_741_824.0
        );

        let device_arc = Arc::new(device.clone());
        let queue_arc = Arc::new(queue);
        let profiler = GpuProfiler::new(&device, 4096, timestamps);
        let uniform_arena = GpuUniformBuffer::new(&device, 1024 * 1024 * 4, "global_uniforms");

        Ok(Arc::new(Self {
            device: device_arc,
            queue: queue_arc,
            adapter,
            memory_info,
            allocated_memory: AtomicUsize::new(0),
            inference_pool: OnceLock::new(),
            profiler,
            uniform_arena,
        }))
    }

    pub fn get_inference_pool(self: &Arc<Self>) -> Arc<Mutex<GpuTensorPool>> {
        self.inference_pool
            .get_or_init(|| {
                log::debug!("creating shared inference pool");
                Arc::new(Mutex::new(GpuTensorPool::new(self.clone())))
            })
            .clone()
    }

    fn calculate_memory_info(adapter: &Adapter, config: &GpuConfig) -> Result<GpuMemoryInfo> {
        let limits = adapter.limits();
        let available_memory = Self::query_gpu_memory(adapter);
        let kv_cache_reservation = config.kv_cache_memory_mb * 1_048_576;

        log::debug!(
            "adapter limits: max_buffer={:.2}GB, max_binding={:.2}GB",
            limits.max_buffer_size as f64 / 1_073_741_824.0,
            limits.max_storage_buffer_binding_size as f64 / 1_073_741_824.0
        );

        if let Some(available) = available_memory {
            log::info!("detected vram: {:.2}GB", available as f64 / 1_073_741_824.0);
        } else {
            log::warn!("could not query available vram");
        }

        Ok(GpuMemoryInfo {
            max_buffer_size: limits.max_buffer_size,
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            available_memory,
            reserved_for_kv_cache: kv_cache_reservation,
        })
    }

    #[cfg(target_os = "windows")]
    fn query_gpu_memory(_adapter: &Adapter) -> Option<u64> {
        // use windows::core::Interface;
        // use windows::Win32::Graphics::Dxgi::*;

        // unsafe {
        //     if let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() {
        //         for i in 0..16 {
        //             if let Ok(dxgi_adapter) = factory.EnumAdapters1(i) {
        //                 let mut video_memory_info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        //                 if dxgi_adapter
        //                     .QueryVideoMemoryInfo(
        //                         0,
        //                         DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        //                         &mut video_memory_info,
        //                     )
        //                     .is_ok()
        //                 {
        //                     if video_memory_info.Budget > 0 {
        //                         return Some(video_memory_info.Budget);
        //                     }
        //                 }
        //             } else {
        //                 break;
        //             }
        //         }
        //     }
        // }

        // log::warn!("failed to query gpu memory via dxgi");
        None
    }

    #[cfg(target_os = "linux")]
    fn query_gpu_memory(_adapter: &Adapter) -> Option<u64> {
        #[cfg(feature = "nvidia")]
        {
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                if let Ok(device) = nvml.device_by_index(0) {
                    if let Ok(memory_info) = device.memory_info() {
                        return Some(memory_info.free);
                    }
                }
            }
        }

        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("card"))
                    .unwrap_or(false)
                {
                    let mem_info_path = path.join("device/mem_info_vram_total");
                    if let Ok(contents) = std::fs::read_to_string(&mem_info_path)
                        && let Ok(bytes) = contents.trim().parse::<u64>()
                    {
                        return Some(bytes);
                    }
                }
            }
        }

        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .output()
            && let Ok(text) = String::from_utf8(output.stdout)
            && let Ok(mb) = text.trim().parse::<u64>()
        {
            return Some(mb * 1_048_576);
        }

        log::warn!("failed to query gpu memory on linux");
        None
    }

    #[cfg(target_os = "macos")]
    fn query_gpu_memory(_adapter: &Adapter) -> Option<u64> {
        use sysinfo::System;

        let mut sys = System::new_all();
        sys.refresh_memory();

        let total_memory = sys.total_memory() * 1024;
        let gpu_share = (total_memory * 3) / 4;

        Some(gpu_share)
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    fn query_gpu_memory(_adapter: &Adapter) -> Option<u64> {
        log::warn!("gpu memory query not implemented for this platform");
        None
    }

    pub fn track_allocation(&self, bytes: usize) {
        self.allocated_memory.fetch_add(bytes, Ordering::Relaxed);
        log::debug!(
            "allocated {} bytes, total: {:.2}MB",
            bytes,
            self.allocated_memory.load(Ordering::Relaxed) as f64 / 1_048_576.0
        );
    }

    pub fn track_deallocation(&self, bytes: usize) {
        self.allocated_memory.fetch_sub(bytes, Ordering::Relaxed);
        log::debug!(
            "deallocated {} bytes, total: {:.2}MB",
            bytes,
            self.allocated_memory.load(Ordering::Relaxed) as f64 / 1_048_576.0
        );
    }

    pub fn get_allocated_memory(&self) -> usize {
        self.allocated_memory.load(Ordering::Relaxed)
    }

    pub fn print_memory_usage(&self) {
        let allocated = self.get_allocated_memory();
        log::info!(
            "gpu memory: {:.2}MB allocated",
            allocated as f64 / 1_048_576.0
        );

        if let Some(available) = self.memory_info.available_memory {
            let percent = (allocated as f64 / available as f64) * 100.0;
            log::info!("vram usage: {:.1}%", percent);
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub kv_cache_memory_mb: u64,
    pub prefer_texture_embeddings: bool,
    pub min_batch_size_for_gpu: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            kv_cache_memory_mb: 1024,
            prefer_texture_embeddings: true,
            min_batch_size_for_gpu: 128,
        }
    }
}

/// Drive the device until queued work completes.
///
/// Native wgpu needs an explicit pump before a `map_async` callback can run. On
/// wasm the browser's event loop owns that, and blocking is not permitted, so
/// this is a no-op and the caller's `.await` is what makes progress.
pub fn drain(device: &wgpu::Device) {
    #[cfg(not(target_arch = "wasm32"))]
    match device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => log::debug!("GPU poll ok: {:?}", status),
        Err(e) => panic!("GPU poll failed: {:?}", e),
    }
    #[cfg(target_arch = "wasm32")]
    let _ = device;
}
