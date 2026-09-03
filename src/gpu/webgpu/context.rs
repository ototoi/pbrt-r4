use crate::util::error::PbrtError;

pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Context {
    pub fn new() -> Result<Self, PbrtError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|error| {
                PbrtError::error(&format!("Could not request a WebGPU adapter: {error}"))
            })?;

        if !adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        {
            return Err(PbrtError::error(
                "The selected WebGPU adapter does not support experimental ray queries.",
            ));
        }

        let adapter_limits = adapter.limits();
        let mut required_limits =
            wgpu::Limits::default().using_minimum_supported_acceleration_structure_values();
        // Keep the default limits for portability, but do not unnecessarily cap
        // large scene buffers at wgpu's conservative default binding size.
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("pbrt-r4 primary-ray device"),
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits,
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor)).map_err(|error| {
                PbrtError::error(&format!("Could not request a WebGPU device: {error}"))
            })?;
        Ok(Self { device, queue })
    }

    pub fn wait(&self) -> Result<(), PbrtError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| PbrtError::error(&format!("WebGPU device polling failed: {error}")))
    }
}
