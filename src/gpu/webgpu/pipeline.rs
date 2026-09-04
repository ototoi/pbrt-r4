use crate::util::error::PbrtError;

use super::shader;

pub struct Pipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub generate_primary_rays: wgpu::ComputePipeline,
    pub intersect_primary_rays: wgpu::ComputePipeline,
    pub handle_escaped: wgpu::ComputePipeline,
    pub prepare_sample: wgpu::ComputePipeline,
    pub shade_surface: wgpu::ComputePipeline,
    pub handle_emissive: wgpu::ComputePipeline,
    pub evaluate_materials: wgpu::ComputePipeline,
    pub intersect_shadow: wgpu::ComputePipeline,
    pub finish_shadow: wgpu::ComputePipeline,
    pub sample_diffuse_bounce: wgpu::ComputePipeline,
    pub swap_ray_queues: wgpu::ComputePipeline,
    pub reset_next_ray_queue: wgpu::ComputePipeline,
    pub reset_shadow_queue: wgpu::ComputePipeline,
    pub reset_classification_queues: wgpu::ComputePipeline,
    pub accumulate_sample: wgpu::ComputePipeline,
}

impl Pipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, PbrtError> {
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbrt-r4 primary-ray bind group layout"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Uniform),
                buffer_entry(1, wgpu::BufferBindingType::Uniform),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                },
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, true),
                storage_entry(8, false),
                storage_entry(9, false),
                storage_entry(10, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbrt-r4 primary-ray pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let compute = |label, stage_source, entry_point| {
            let shader = shader::create_module(device, label, stage_source);
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
                include_str!("shaders/generate_primary_rays.wgsl"),
                "generate_primary_rays",
            ),
            intersect_primary_rays: compute(
                "pbrt-r4 intersect primary rays",
                include_str!("shaders/intersect_primary_rays.wgsl"),
                "intersect_primary_rays",
            ),
            handle_escaped: compute(
                "pbrt-r4 handle escaped rays",
                include_str!("shaders/handle_escaped.wgsl"),
                "handle_escaped",
            ),
            prepare_sample: compute(
                "pbrt-r4 prepare sample",
                include_str!("shaders/prepare_sample.wgsl"),
                "prepare_sample",
            ),
            shade_surface: compute(
                "pbrt-r4 shade surface",
                include_str!("shaders/shade_surface.wgsl"),
                "shade_surface",
            ),
            handle_emissive: compute(
                "pbrt-r4 handle emissive",
                include_str!("shaders/handle_emissive.wgsl"),
                "handle_emissive",
            ),
            evaluate_materials: compute(
                "pbrt-r4 evaluate materials",
                include_str!("shaders/evaluate_materials.wgsl"),
                "evaluate_materials",
            ),
            intersect_shadow: compute(
                "pbrt-r4 intersect shadow",
                include_str!("shaders/intersect_shadow.wgsl"),
                "intersect_shadow",
            ),
            finish_shadow: compute(
                "pbrt-r4 finish shadow",
                include_str!("shaders/finish_shadow.wgsl"),
                "finish_shadow",
            ),
            sample_diffuse_bounce: compute(
                "pbrt-r4 sample diffuse bounce",
                include_str!("shaders/sample_diffuse_bounce.wgsl"),
                "sample_diffuse_bounce",
            ),
            swap_ray_queues: compute(
                "pbrt-r4 swap ray queues",
                include_str!("shaders/swap_ray_queues.wgsl"),
                "swap_ray_queues",
            ),
            reset_next_ray_queue: compute(
                "pbrt-r4 reset next ray queue",
                include_str!("shaders/reset_next_ray_queue.wgsl"),
                "reset_next_ray_queue",
            ),
            reset_shadow_queue: compute(
                "pbrt-r4 reset shadow queue",
                include_str!("shaders/reset_shadow_queue.wgsl"),
                "reset_shadow_queue",
            ),
            reset_classification_queues: compute(
                "pbrt-r4 reset classification queues",
                include_str!("shaders/reset_classification_queues.wgsl"),
                "reset_classification_queues",
            ),
            accumulate_sample: compute(
                "pbrt-r4 accumulate sample",
                include_str!("shaders/accumulate_sample.wgsl"),
                "accumulate_sample",
            ),
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
