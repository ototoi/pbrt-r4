use crate::util::error::PbrtError;

pub struct Pipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub generate_primary_rays: wgpu::ComputePipeline,
    pub intersect_primary_rays: wgpu::ComputePipeline,
    pub shade_normal: wgpu::ComputePipeline,
}

impl Pipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, PbrtError> {
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbrt-r4 primary-ray bind group layout"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Uniform),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                },
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, false),
                storage_entry(8, false),
                storage_entry(9, false),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbrt-r4 primary-ray-normal shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/primary_ray_normal.wgsl"
            ))),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbrt-r4 primary-ray pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let compute = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline = Self {
            bind_group_layout,
            generate_primary_rays: compute(
                "pbrt-r4 generate primary rays",
                "generate_primary_rays",
            ),
            intersect_primary_rays: compute(
                "pbrt-r4 intersect primary rays",
                "intersect_primary_rays",
            ),
            shade_normal: compute("pbrt-r4 shade normal", "shade_normal"),
        };
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(PbrtError::error(&format!(
                "WebGPU primary-ray pipeline creation failed: {error}"
            )));
        }
        Ok(pipeline)
    }
}

fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    buffer_entry(binding, wgpu::BufferBindingType::Storage { read_only })
}
