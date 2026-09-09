use crate::util::error::PbrtError;

use super::stages::RequiredLimits;

pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Context {
    pub fn new(
        required: RequiredLimits,
        texture_image_count: u32,
        texture_sampler_count: u32,
    ) -> Result<Self, PbrtError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|error| {
                PbrtError::error(&format!("Could not request a WebGPU adapter: {error}"))
            })?;
        log::info!("GPU context: adapter selected: {:?}", adapter.get_info());

        let required_features = wgpu::Features::EXPERIMENTAL_RAY_QUERY
            | wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::FLOAT32_FILTERABLE;
        let missing_features = required_features - adapter.features();
        if !missing_features.is_empty() {
            return Err(PbrtError::error(&format!(
                "The selected WebGPU adapter is missing required features: {missing_features:?}.",
            )));
        }

        let adapter_limits = adapter.limits();
        if required.storage_buffers_per_shader_stage
            > adapter_limits.max_storage_buffers_per_shader_stage
            || required.uniform_buffers_per_shader_stage
                > adapter_limits.max_uniform_buffers_per_shader_stage
            || required.bind_groups > adapter_limits.max_bind_groups
        {
            return Err(PbrtError::error(&format!(
                "WebGPU adapter limits are insufficient: requested storage={}, uniform={}, bind_groups={}; available storage={}, uniform={}, bind_groups={}",
                required.storage_buffers_per_shader_stage,
                required.uniform_buffers_per_shader_stage,
                required.bind_groups,
                adapter_limits.max_storage_buffers_per_shader_stage,
                adapter_limits.max_uniform_buffers_per_shader_stage,
                adapter_limits.max_bind_groups,
            )));
        }
        if texture_image_count > adapter_limits.max_binding_array_elements_per_shader_stage
            || texture_sampler_count
                > adapter_limits.max_binding_array_sampler_elements_per_shader_stage
        {
            return Err(PbrtError::error(
                &format!(
                    "WebGPU adapter supports texture/sampler binding arrays of {}/{} elements, but the scene requires {}/{}.",
                    adapter_limits.max_binding_array_elements_per_shader_stage,
                    adapter_limits.max_binding_array_sampler_elements_per_shader_stage,
                    texture_image_count,
                    texture_sampler_count,
                ),
            ));
        }
        let mut required_limits =
            wgpu::Limits::default().using_minimum_supported_acceleration_structure_values();
        // Keep the default limits for portability, but do not unnecessarily cap
        // large scene buffers at wgpu's conservative default binding size.
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_sampled_textures_per_shader_stage =
            adapter_limits.max_sampled_textures_per_shader_stage;
        required_limits.max_samplers_per_shader_stage =
            adapter_limits.max_samplers_per_shader_stage;
        required_limits.max_binding_array_elements_per_shader_stage =
            adapter_limits.max_binding_array_elements_per_shader_stage;
        required_limits.max_binding_array_sampler_elements_per_shader_stage =
            adapter_limits.max_binding_array_sampler_elements_per_shader_stage;
        required_limits.max_storage_buffers_per_shader_stage =
            required.storage_buffers_per_shader_stage;
        required_limits.max_uniform_buffers_per_shader_stage =
            required.uniform_buffers_per_shader_stage;
        required_limits.max_bind_groups = required.bind_groups;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("pbrt-r4 primary-ray device"),
            required_features,
            required_limits,
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor)).map_err(|error| {
                PbrtError::error(&format!("Could not request a WebGPU device: {error}"))
            })?;
        log::info!("GPU context: device created");
        Ok(Self { device, queue })
    }

    pub fn wait(&self) -> Result<(), PbrtError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| PbrtError::error(&format!("WebGPU device polling failed: {error}")))
    }
}
