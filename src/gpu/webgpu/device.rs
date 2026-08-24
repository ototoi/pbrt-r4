use super::error::BackendError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PowerPreference {
    LowPower,
    HighPerformance,
}

impl Default for PowerPreference {
    fn default() -> Self {
        Self::HighPerformance
    }
}

impl From<PowerPreference> for wgpu::PowerPreference {
    fn from(preference: PowerPreference) -> Self {
        match preference {
            PowerPreference::LowPower => Self::LowPower,
            PowerPreference::HighPerformance => Self::HighPerformance,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccelerationMode {
    HardwareRayQuery,
    SoftwareBvh,
}

impl Default for AccelerationMode {
    fn default() -> Self {
        Self::HardwareRayQuery
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrepareOptions {
    pub power_preference: PowerPreference,
    pub force_fallback_adapter: bool,
    pub acceleration_mode: AccelerationMode,
    pub max_texture_dimension_2d: Option<u32>,
}

pub struct DeviceContext {
    _instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub acceleration_mode: AccelerationMode,
    pub max_texture_dimension_2d: u32,
}

impl DeviceContext {
    pub fn create(options: &PrepareOptions) -> Result<Self, BackendError> {
        if options.max_texture_dimension_2d == Some(0) {
            return Err(BackendError::InvalidPrepareOptions {
                reason: "max_texture_dimension_2d must be greater than zero",
            });
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: options.power_preference.into(),
            force_fallback_adapter: options.force_fallback_adapter,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|error| BackendError::AdapterRequest(error.to_string()))?;

        let adapter_max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
        let max_texture_dimension_2d = options
            .max_texture_dimension_2d
            .unwrap_or(adapter_max_texture_dimension_2d)
            .min(adapter_max_texture_dimension_2d);
        let (required_features, required_limits, experimental_features) = match options
            .acceleration_mode
        {
            AccelerationMode::HardwareRayQuery => {
                if !adapter
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                {
                    return Err(BackendError::MissingRayQueryFeature);
                }
                (
                    wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                    wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(),
                    // SAFETY: hardware mode is only requested after the adapter capability
                    // check above. The experimental API is explicitly part of this backend's
                    // selected mode.
                    unsafe { wgpu::ExperimentalFeatures::enabled() },
                )
            }
            AccelerationMode::SoftwareBvh => (
                wgpu::Features::empty(),
                wgpu::Limits::default(),
                wgpu::ExperimentalFeatures::disabled(),
            ),
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pbrt-r4 webgpu device"),
            required_features,
            required_limits,
            experimental_features,
            ..Default::default()
        }))
        .map_err(|error| BackendError::DeviceRequest(error.to_string()))?;

        Ok(Self {
            _instance: instance,
            adapter,
            device,
            queue,
            acceleration_mode: options.acceleration_mode,
            max_texture_dimension_2d,
        })
    }
}
