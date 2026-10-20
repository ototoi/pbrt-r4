use crate::util::error::PbrtError;

use super::binding_report::{BindingHeadroom, BindingReport};
use super::stages::RequiredLimits;

pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub struct AdapterContext {
    pub _instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
}

impl Context {
    pub fn new(
        required: RequiredLimits,
        texture_image_count: u32,
        texture_sampler_count: u32,
    ) -> Result<Self, PbrtError> {
        Self::create(
            required,
            texture_image_count,
            texture_sampler_count,
            1,
            None,
        )
    }

    pub fn new_with_geometry_page_count(
        required: RequiredLimits,
        texture_image_count: u32,
        texture_sampler_count: u32,
        geometry_page_count: u32,
    ) -> Result<Self, PbrtError> {
        Self::create(
            required,
            texture_image_count,
            texture_sampler_count,
            geometry_page_count,
            None,
        )
    }

    pub fn with_binding_report(
        report: &BindingReport,
        texture_image_count: u32,
        texture_sampler_count: u32,
    ) -> Result<Self, PbrtError> {
        Self::create(
            report.device_required_for_geometry_pages(1),
            texture_image_count,
            texture_sampler_count,
            1,
            Some(report),
        )
    }

    pub fn with_binding_report_and_geometry_page_count(
        report: &BindingReport,
        texture_image_count: u32,
        texture_sampler_count: u32,
        geometry_page_count: u32,
    ) -> Result<Self, PbrtError> {
        Self::create(
            report.device_required_for_geometry_pages(geometry_page_count),
            texture_image_count,
            texture_sampler_count,
            geometry_page_count,
            Some(report),
        )
    }

    pub fn request_adapter(report: Option<&BindingReport>) -> Result<AdapterContext, PbrtError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|error| {
                PbrtError::error(&format!(
                    "Could not request a WebGPU adapter: {error}{}",
                    report
                        .map(|report| format!("\n{}", report.format(None, false)))
                        .unwrap_or_default()
                ))
            })?;
        let info = adapter.get_info();
        log::info!("GPU context: adapter selected: {info:?}");
        let limits = adapter.limits();
        let features = adapter.features();
        if log::log_enabled!(log::Level::Debug) {
            if let Some(report) = report {
                log::debug!("adapter: {info:?}\n{}", report.format(Some(&limits), false));
            }
        }
        Ok(AdapterContext {
            _instance: instance,
            adapter,
            info,
            limits,
            features,
        })
    }

    pub fn from_adapter(
        adapter: AdapterContext,
        mut required: RequiredLimits,
        texture_image_count: u32,
        texture_sampler_count: u32,
        geometry_page_count: u32,
        subsurface_enabled: bool,
        report: Option<&BindingReport>,
    ) -> Result<Self, PbrtError> {
        let mut required_features = wgpu::Features::EXPERIMENTAL_RAY_QUERY
            | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
            | wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::FLOAT32_FILTERABLE;
        if geometry_page_count > 1 {
            required_features |= wgpu::Features::BUFFER_BINDING_ARRAY;
        } else if let Some(report) = report {
            required = required.maximum(report.device_required_for_geometry_pages(1));
        } else {
            // The one-page shader variant binds vertices and indices as
            // ordinary storage buffers, so count them against buffer limits.
            required.storage_buffers_per_shader_stage =
                required.storage_buffers_per_shader_stage.saturating_add(2);
            required.buffers_and_acceleration_structures_per_shader_stage = required
                .buffers_and_acceleration_structures_per_shader_stage
                .saturating_add(2);
        }
        let missing_features = required_features - adapter.features;
        if !missing_features.is_empty() {
            return Err(PbrtError::error(&format!(
                "The selected WebGPU adapter {:?} is missing required features: {missing_features:?}.",
                adapter.info
            )));
        }
        let (array_elements, sampler_elements, metal_slots) = report
            .map(|report| {
                report.binding_array_usage(
                    geometry_page_count,
                    texture_image_count,
                    texture_sampler_count,
                    subsurface_enabled,
                )
            })
            .transpose()?
            .unwrap_or((
                texture_image_count.saturating_add(if geometry_page_count > 1 {
                    geometry_page_count.saturating_mul(2)
                } else {
                    0
                }),
                texture_sampler_count,
                required
                    .buffers_and_acceleration_structures_per_shader_stage
                    .saturating_add(if geometry_page_count == 1 { 2 } else { 4 }),
            ));
        let array_limit = u64::from(adapter.limits.max_binding_array_elements_per_shader_stage);
        let sampler_limit = adapter
            .limits
            .max_binding_array_sampler_elements_per_shader_stage;
        if u64::from(array_elements) > array_limit || sampler_elements > sampler_limit {
            return Err(PbrtError::error(&format!(
                "WebGPU adapter {:?} binding arrays exceed limits: geometry pages={geometry_page_count} (2 arrays), texture images={texture_image_count}, texture samplers={texture_sampler_count}; total elements {array_elements}/{array_limit}, sampler elements {sampler_elements}/{sampler_limit}.",
                adapter.info
            )));
        }
        if adapter.info.backend == wgpu::Backend::Metal
            && metal_slots
                > adapter
                    .limits
                    .max_buffers_and_acceleration_structures_per_shader_stage
        {
            return Err(PbrtError::error(&format!(
                "WebGPU Metal adapter {:?} requires {metal_slots} buffer argument slots per stage, exceeding the limit {}.",
                adapter.info,
                adapter.limits.max_buffers_and_acceleration_structures_per_shader_stage,
            )));
        }
        validate_binding_limits(required, &adapter.limits, report)
            .map_err(|error| PbrtError::error(&format!("{error}\nadapter: {:?}", adapter.info)))?;
        let mut required_limits =
            wgpu::Limits::default().using_minimum_supported_acceleration_structure_values();
        required_limits.max_buffer_size = adapter.limits.max_buffer_size;
        required_limits.max_storage_buffer_binding_size =
            adapter.limits.max_storage_buffer_binding_size;
        required_limits.max_sampled_textures_per_shader_stage =
            adapter.limits.max_sampled_textures_per_shader_stage;
        required_limits.max_samplers_per_shader_stage =
            adapter.limits.max_samplers_per_shader_stage;
        required_limits.max_binding_array_elements_per_shader_stage =
            adapter.limits.max_binding_array_elements_per_shader_stage;
        required_limits.max_binding_array_sampler_elements_per_shader_stage = adapter
            .limits
            .max_binding_array_sampler_elements_per_shader_stage;
        required_limits.max_storage_buffers_per_shader_stage =
            required.storage_buffers_per_shader_stage;
        required_limits.max_uniform_buffers_per_shader_stage =
            required.uniform_buffers_per_shader_stage;
        required_limits.max_buffers_and_acceleration_structures_per_shader_stage =
            required.buffers_and_acceleration_structures_per_shader_stage;
        required_limits.max_bind_groups = required.bind_groups;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("pbrt-r4 primary-ray device"),
            required_features,
            required_limits,
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        let (device, queue) = pollster::block_on(adapter.adapter.request_device(&descriptor))
            .map_err(|error| {
                PbrtError::error(&format!(
                    "Could not request a WebGPU device: {error}{}",
                    report
                        .map(|report| format!(
                            "\nadapter: {:?}\n{}",
                            adapter.info,
                            report.format(Some(&adapter.limits), false)
                        ))
                        .unwrap_or_default()
                ))
            })?;
        log::info!("GPU context: device created");
        Ok(Self { device, queue })
    }

    fn create(
        required: RequiredLimits,
        texture_image_count: u32,
        texture_sampler_count: u32,
        geometry_page_count: u32,
        report: Option<&BindingReport>,
    ) -> Result<Self, PbrtError> {
        let adapter = Self::request_adapter(report)?;
        Self::from_adapter(
            adapter,
            required,
            texture_image_count,
            texture_sampler_count,
            geometry_page_count,
            false,
            report,
        )
    }

    pub fn wait(&self) -> Result<(), PbrtError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| PbrtError::error(&format!("WebGPU device polling failed: {error}")))
    }
}

/// Validate buffer requirements against adapter limits without creating a device.
/// Includes over-budget stage details when a binding report is provided.
pub fn validate_binding_limits(
    required: RequiredLimits,
    available: &wgpu::Limits,
    report: Option<&BindingReport>,
) -> Result<(), PbrtError> {
    if BindingHeadroom::new(required, available).fits() {
        return Ok(());
    }
    Err(PbrtError::error(&format!(
        "WebGPU adapter limits are insufficient: requested storage={}, uniform={}, buffers_and_acceleration_structures={}, bind_groups={}; available storage={}, uniform={}, buffers_and_acceleration_structures={}, bind_groups={}{}",
        required.storage_buffers_per_shader_stage,
        required.uniform_buffers_per_shader_stage,
        required.buffers_and_acceleration_structures_per_shader_stage,
        required.bind_groups,
        available.max_storage_buffers_per_shader_stage,
        available.max_uniform_buffers_per_shader_stage,
        available.max_buffers_and_acceleration_structures_per_shader_stage,
        available.max_bind_groups,
        report.map(|report| format!("\n{}", report.format(Some(available), true))).unwrap_or_default(),
    )))
}
