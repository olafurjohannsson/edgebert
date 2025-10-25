use wgpu::{
    DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits, PowerPreference,
    RequestAdapterOptions,
};

pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuContext {
    pub async fn new() -> Self {
        println!("=== Initializing WGPU ===");

        let instance = Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::empty(), // Make sure this is empty, NOT debugging
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Log adapter info
        let info = adapter.get_info();
        println!("Selected GPU: {}", info.name);
        println!("  Backend: {:?}", info.backend);
        println!("  Device Type: {:?}", info.device_type);
        println!("  Vendor: {:?}", info.vendor);
        println!("  Driver: {}", info.driver);

        // Get adapter limits to see what's available
        let adapter_limits = adapter.limits();
        println!("Adapter Limits:");
        println!(
            "  Max buffer size: {} MB",
            adapter_limits.max_buffer_size / 1_048_576
        );
        println!(
            "  Max compute workgroup size: {}x{}x{}",
            adapter_limits.max_compute_workgroup_size_x,
            adapter_limits.max_compute_workgroup_size_y,
            adapter_limits.max_compute_workgroup_size_z
        );

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("EdgeGPT"),
                required_features: Features::TIMESTAMP_QUERY
                    | Features::TIMESTAMP_QUERY_INSIDE_PASSES,
                required_limits: Limits {
                    max_compute_workgroup_size_x: 1024,
                    max_compute_workgroup_size_y: 1024,
                    max_compute_invocations_per_workgroup: 1024,
                    max_buffer_size: adapter_limits.max_buffer_size.min(1_073_741_824), // Request min of adapter limit or 1GB
                    ..Limits::default()
                },
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        println!("Device created successfully");
        println!(
            "  Max buffer size: {} MB",
            device.limits().max_buffer_size / 1_048_576
        );
        println!("=========================\n");

        Self { device, queue }
    }
}
