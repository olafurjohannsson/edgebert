use wgpu::{
    InstanceDescriptor, DeviceDescriptor, Features, Limits, RequestAdapterOptions, Instance};

pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuContext {
    pub async fn new() -> Self {
        let instance = Instance::new(&InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                // The second argument has been removed from the function signature.
            )
            .await
            .expect("Failed to create device");
        
        Self { device, queue }
    }
}