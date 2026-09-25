use crate::util::error::PbrtError;

use super::shader;
use super::stages::{
    all_stage_specs, canonical_wavefront_bindings, BindingClass, BindingSpec, RequiredLimits,
    ResourceId,
};

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
    pub prepare_queue_dispatch: StagePipeline,
    pub evaluate_textures: StagePipeline,
    pub evaluate_attributes: StagePipeline,
    pub classify_surface_scatter: StagePipeline,
    pub select_portal_direct: StagePipeline,
    pub sample_portal_direct: StagePipeline,
    pub sample_direct_light: StagePipeline,
    pub scatter_diffuse: StagePipeline,
    pub scatter_diffuse_transmission: StagePipeline,
    pub scatter_conductor: StagePipeline,
    pub scatter_dielectric: StagePipeline,
    pub scatter_thin_dielectric: StagePipeline,
    pub scatter_measured: StagePipeline,
    pub scatter_coated: StagePipeline,
    pub intersect_shadow: StagePipeline,
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
        texture_program_capacity: u32,
        texture_noise_enabled: bool,
    ) -> Result<Self, PbrtError> {
        // Validate the complete stage contract before creating the deployed
        // layout. The canonical registry supplies the current ABI entries.
        RequiredLimits::from_stages(&all_stage_specs())?;
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let canonical_bindings = canonical_wavefront_bindings();
        let compute = |label: &'static str,
                       stage_source: &'static str,
                       entry_point: &'static str| {
            log::info!("GPU pipeline: creating {label}");
            let source = shader::compose_source_with_noise(stage_source, texture_noise_enabled)
                .replace(
                    "const TEXTURE_PROGRAM_CAPACITY: u32 = 256u;",
                    &format!("const TEXTURE_PROGRAM_CAPACITY: u32 = {texture_program_capacity}u;"),
                )
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
            prepare_queue_dispatch: compute(
                "pbrt-r4 prepare queue dispatch",
                include_str!("shaders/prepare_queue_dispatch.wgsl"),
                "prepare_queue_dispatch",
            ),
            evaluate_textures: compute(
                "pbrt-r4 evaluate textures",
                include_str!("shaders/evaluate_textures.wgsl"),
                "evaluate_textures",
            ),
            evaluate_attributes: compute(
                "pbrt-r4 evaluate attributes",
                include_str!("shaders/evaluate_attributes.wgsl"),
                "evaluate_attributes",
            ),
            classify_surface_scatter: compute(
                "pbrt-r4 classify surface scatter",
                include_str!("shaders/classify_surface_scatter.wgsl"),
                "classify_surface_scatter",
            ),
            select_portal_direct: compute(
                "pbrt-r4 select portal direct",
                include_str!("shaders/select_portal_direct.wgsl"),
                "select_portal_direct",
            ),
            sample_portal_direct: compute(
                "pbrt-r4 sample portal direct",
                include_str!("shaders/sample_portal_direct.wgsl"),
                "sample_portal_direct",
            ),
            sample_direct_light: compute(
                "pbrt-r4 sample direct light",
                include_str!("shaders/sample_direct_light.wgsl"),
                "sample_direct_light",
            ),
            scatter_diffuse: compute(
                "pbrt-r4 scatter diffuse",
                include_str!("shaders/scatter_diffuse.wgsl"),
                "scatter_diffuse",
            ),
            scatter_diffuse_transmission: compute(
                "pbrt-r4 scatter diffuse transmission",
                include_str!("shaders/scatter_diffuse_transmission.wgsl"),
                "scatter_diffuse_transmission",
            ),
            scatter_conductor: compute(
                "pbrt-r4 scatter conductor",
                include_str!("shaders/scatter_conductor.wgsl"),
                "scatter_conductor",
            ),
            scatter_dielectric: compute(
                "pbrt-r4 scatter dielectric",
                include_str!("shaders/scatter_dielectric.wgsl"),
                "scatter_dielectric",
            ),
            scatter_thin_dielectric: compute(
                "pbrt-r4 scatter thin dielectric",
                include_str!("shaders/scatter_thin_dielectric.wgsl"),
                "scatter_thin_dielectric",
            ),
            scatter_measured: compute(
                "pbrt-r4 scatter measured",
                include_str!("shaders/scatter_measured.wgsl"),
                "scatter_measured",
            ),
            scatter_coated: compute(
                "pbrt-r4 scatter coated",
                include_str!("shaders/scatter_coated.wgsl"),
                "scatter_coated",
            ),
            intersect_shadow: compute(
                "pbrt-r4 intersect shadow",
                include_str!("shaders/intersect_shadow.wgsl"),
                "intersect_shadow",
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
    binding: &BindingSpec,
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
        BindingClass::IntegerTexture => wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
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
            ResourceId::TextureImageArray | ResourceId::TextureSamplerArray
        ) {
            let count = match binding.resource {
                ResourceId::TextureImageArray => texture_image_count,
                ResourceId::TextureSamplerArray => texture_sampler_count,
                _ => unreachable!("only texture arrays have a binding count"),
            };
            std::num::NonZeroU32::new(count.max(1))
        } else {
            None
        },
    }
}
