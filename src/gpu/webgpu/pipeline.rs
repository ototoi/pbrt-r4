use crate::util::error::PbrtError;

use super::shader;
use super::stages::{all_stage_specs, canonical_wavefront_bindings, BindingClass, RequiredLimits};

pub struct StagePipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

pub struct Pipeline {
    pub generate_primary_rays: StagePipeline,
    pub intersect_primary_rays: StagePipeline,
    pub handle_escaped: StagePipeline,
    pub prepare_sample: StagePipeline,
    pub shade_surface: StagePipeline,
    pub handle_emissive: StagePipeline,
    pub evaluate_materials: StagePipeline,
    pub intersect_shadow: StagePipeline,
    pub sample_diffuse_bounce: StagePipeline,
    pub sample_dielectric_bounce: StagePipeline,
    pub sample_conductor_bounce: StagePipeline,
    pub sample_thin_dielectric_bounce: StagePipeline,
    pub sample_composite_bounce: StagePipeline,
    pub swap_ray_queues: StagePipeline,
    pub reset_next_ray_queue: StagePipeline,
    pub reset_shadow_queue: StagePipeline,
    pub reset_classification_queues: StagePipeline,
    pub accumulate_sample: StagePipeline,
}

impl Pipeline {
    pub fn new(
        device: &wgpu::Device,
        texture_image_count: u32,
        texture_sampler_count: u32,
    ) -> Result<Self, PbrtError> {
        // Validate the complete stage contract before creating the deployed
        // layout. The canonical registry supplies the current ABI entries.
        RequiredLimits::from_stages(&all_stage_specs())?;
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let canonical_bindings = canonical_wavefront_bindings();
        let compute =
            |label: &'static str, stage_source: &'static str, entry_point: &'static str| {
                log::info!("GPU pipeline: creating {label}");
                let source = shader::compose_source(stage_source)
                    .replace(
                        "binding_array<texture_2d<f32>>",
                        &format!("binding_array<texture_2d<f32>, {texture_image_count}u>"),
                    )
                    .replace(
                        "binding_array<sampler>",
                        &format!("binding_array<sampler, {texture_sampler_count}u>"),
                    );
                log::info!(
                    "GPU pipeline: {label} source composed ({} bytes)",
                    source.len()
                );
                let used_bindings = shader::resource_bindings(&source);
                let bind_group_layouts = (0..=1)
                    .map(|group| {
                        let layout_entries = canonical_bindings
                            .iter()
                            .filter(|binding| {
                                binding.group == group
                                    && used_bindings.contains(&(binding.group, binding.binding))
                            })
                            .map(|binding| {
                                layout_entry(binding, texture_image_count, texture_sampler_count)
                            })
                            .collect::<Vec<_>>();
                        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some(label),
                            entries: &layout_entries,
                        })
                    })
                    .collect::<Vec<_>>();
                log::info!("GPU pipeline: {label} bind group layouts created");
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: bind_group_layouts
                        .iter()
                        .map(Some)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    immediate_size: 0,
                });
                log::info!("GPU pipeline: {label} pipeline layout created");
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(source)),
                });
                log::info!("GPU pipeline: {label} shader module created");
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });
                log::info!("GPU pipeline: created {label}");
                StagePipeline {
                    pipeline,
                    bind_group_layouts,
                }
            };
        let pipeline = Self {
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
            sample_diffuse_bounce: compute(
                "pbrt-r4 sample diffuse bounce",
                include_str!("shaders/sample_diffuse_bounce.wgsl"),
                "sample_diffuse_bounce",
            ),
            sample_dielectric_bounce: compute(
                "pbrt-r4 sample dielectric bounce",
                include_str!("shaders/sample_dielectric_bounce.wgsl"),
                "sample_dielectric_bounce",
            ),
            sample_conductor_bounce: compute(
                "pbrt-r4 sample conductor bounce",
                include_str!("shaders/sample_conductor_bounce.wgsl"),
                "sample_conductor_bounce",
            ),
            sample_thin_dielectric_bounce: compute(
                "pbrt-r4 sample thin dielectric bounce",
                include_str!("shaders/sample_thin_dielectric_bounce.wgsl"),
                "sample_thin_dielectric_bounce",
            ),
            sample_composite_bounce: compute(
                "pbrt-r4 sample composite bounce",
                include_str!("shaders/sample_composite_bounce.wgsl"),
                "sample_composite_bounce",
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

fn layout_entry(
    binding: &super::stages::BindingSpec,
    texture_image_count: u32,
    texture_sampler_count: u32,
) -> wgpu::BindGroupLayoutEntry {
    let ty = match binding.class {
        BindingClass::Uniform => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindingClass::Storage => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage {
                read_only: !binding.access.permits_write(),
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindingClass::AccelerationStructure => wgpu::BindingType::AccelerationStructure {
            vertex_return: false,
        },
        BindingClass::SampledTexture => wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        BindingClass::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    };
    wgpu::BindGroupLayoutEntry {
        binding: binding.binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: if matches!(
            binding.resource,
            super::stages::ResourceId::TextureImageArray
                | super::stages::ResourceId::TextureSamplerArray
        ) {
            let count = match binding.resource {
                super::stages::ResourceId::TextureImageArray => texture_image_count,
                super::stages::ResourceId::TextureSamplerArray => texture_sampler_count,
                _ => unreachable!("only texture arrays have a binding count"),
            };
            std::num::NonZeroU32::new(count.max(1))
        } else {
            None
        },
    }
}
