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
pub enum BackendPreference {
    #[default]
    Auto,
    Vulkan,
    Dx12,
    Metal,
}

impl BackendPreference {
    fn backends(self) -> wgpu::Backends {
        match self {
            Self::Auto => target_backends(),
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Dx12 => wgpu::Backends::DX12,
            Self::Metal => wgpu::Backends::METAL,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrepareOptions {
    pub power_preference: PowerPreference,
    pub force_fallback_adapter: bool,
    pub acceleration_mode: AccelerationMode,
    pub max_texture_dimension_2d: Option<u32>,
    pub backend: BackendPreference,
    pub adapter_name: Option<String>,
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
        let (required_features, required_limits, experimental_features) =
            match options.acceleration_mode {
                AccelerationMode::HardwareRayQuery => (
                    wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                    wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(),
                    // SAFETY: hardware mode is only requested after the adapter capability
                    // check above. The experimental API is explicitly part of this backend's
                    // selected mode.
                    unsafe { wgpu::ExperimentalFeatures::enabled() },
                ),
                AccelerationMode::SoftwareBvh => (
                    wgpu::Features::empty(),
                    wgpu::Limits::default(),
                    wgpu::ExperimentalFeatures::disabled(),
                ),
            };

        let backends = options.backend.backends();
        let mut candidates = pollster::block_on(instance.enumerate_adapters(backends));
        let all_candidate_infos = candidates
            .iter()
            .map(|adapter| format_adapter_info(&adapter.get_info()))
            .collect::<Vec<_>>();
        if let Some(adapter_name) = options.adapter_name.as_deref() {
            candidates.retain(|adapter| adapter.get_info().name == adapter_name);
        }
        if options.acceleration_mode == AccelerationMode::HardwareRayQuery {
            let had_feature_candidate = candidates.iter().any(|adapter| {
                adapter
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            });
            candidates.retain(|adapter| {
                adapter.features().contains(required_features)
                    && adapter.limits().check_limits(&required_limits)
            });
            if candidates.is_empty() && had_feature_candidate {
                return Err(BackendError::AdapterRequest(format!(
                    "no adapter satisfies the required ray-query limits; candidates: {}",
                    all_candidate_infos.join(", ")
                )));
            }
            if candidates.is_empty() && options.adapter_name.is_none() {
                return Err(BackendError::MissingRayQueryFeature);
            }
        }
        if candidates.is_empty() && options.force_fallback_adapter {
            let fallback =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: options.power_preference.into(),
                    force_fallback_adapter: true,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                }))
                .map_err(|error| BackendError::AdapterRequest(error.to_string()))?;
            if options
                .adapter_name
                .as_deref()
                .is_none_or(|name| fallback.get_info().name == name)
                && options
                    .backend
                    .backends()
                    .contains(fallback.get_info().backend.into())
                && fallback.features().contains(required_features)
                && (options.acceleration_mode == AccelerationMode::SoftwareBvh
                    || fallback.limits().check_limits(&required_limits))
            {
                candidates.push(fallback);
            }
        }
        if candidates.is_empty() {
            return Err(BackendError::AdapterRequest(format!(
                "no suitable adapter for backend {:?}, adapter name {:?}; candidates: {}",
                options.backend,
                options.adapter_name,
                all_candidate_infos.join(", ")
            )));
        }
        candidates.sort_by(|left, right| {
            adapter_sort_key(&left.get_info(), options.power_preference).cmp(&adapter_sort_key(
                &right.get_info(),
                options.power_preference,
            ))
        });
        let adapter = candidates.remove(0);

        let adapter_max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
        let max_texture_dimension_2d = options
            .max_texture_dimension_2d
            .unwrap_or(adapter_max_texture_dimension_2d)
            .min(adapter_max_texture_dimension_2d);

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

fn target_backends() -> wgpu::Backends {
    #[cfg(windows)]
    {
        wgpu::Backends::DX12 | wgpu::Backends::VULKAN
    }
    #[cfg(target_os = "linux")]
    {
        wgpu::Backends::VULKAN
    }
    #[cfg(target_os = "macos")]
    {
        wgpu::Backends::METAL
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        wgpu::Backends::all()
    }
}

fn adapter_sort_key(info: &wgpu::AdapterInfo, preference: PowerPreference) -> (u8, u8, String) {
    let device_rank = match (preference, info.device_type) {
        (PowerPreference::HighPerformance, wgpu::DeviceType::DiscreteGpu) => 0,
        (PowerPreference::LowPower, wgpu::DeviceType::IntegratedGpu) => 0,
        (_, wgpu::DeviceType::VirtualGpu) => 1,
        (_, wgpu::DeviceType::DiscreteGpu) => 2,
        (_, wgpu::DeviceType::IntegratedGpu) => 3,
        (_, wgpu::DeviceType::Other) => 4,
        (_, wgpu::DeviceType::Cpu) => 5,
    };
    let backend_rank = match info.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Dx12 => 1,
        wgpu::Backend::Metal => 2,
        wgpu::Backend::Gl => 3,
        wgpu::Backend::BrowserWebGpu => 4,
        wgpu::Backend::Noop => 5,
    };
    (device_rank, backend_rank, info.name.clone())
}

fn format_adapter_info(info: &wgpu::AdapterInfo) -> String {
    format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
}
