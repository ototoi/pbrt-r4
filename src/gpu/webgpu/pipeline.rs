use std::collections::HashMap;

use crate::util::error::PbrtError;

use super::shader;
use super::stage::{ComputeStageId, COMPUTE_STAGES};
use super::stages::{canonical_wavefront_bindings, BindingClass, BindingSpec, ResourceId};

pub struct StagePipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    used_bindings: Vec<(u32, u32)>,
}

pub struct Pipeline {
    stages: HashMap<ComputeStageId, StagePipeline>,
}

impl Pipeline {
    pub fn stage(&self, id: ComputeStageId) -> &StagePipeline {
        // Pipeline::new installs every ID emitted by the shared stage definition.
        self.stages
            .get(&id)
            .expect("every declared compute stage has a pipeline")
    }

    pub fn new(
        device: &wgpu::Device,
        texture_image_count: u32,
        texture_sampler_count: u32,
        texture_program_capacity: u32,
        texture_noise_enabled: bool,
    ) -> Result<Self, PbrtError> {
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let canonical_bindings = canonical_wavefront_bindings();
        let stages: HashMap<ComputeStageId, StagePipeline> = COMPUTE_STAGES
            .iter()
            .map(|stage| {
                log::info!("GPU pipeline: creating {}", stage.label);
                let source = shader::compose_source_with_noise(stage.source, texture_noise_enabled)
                    .replace(
                        "const TEXTURE_PROGRAM_CAPACITY: u32 = 256u;",
                        &format!(
                            "const TEXTURE_PROGRAM_CAPACITY: u32 = {texture_program_capacity}u;"
                        ),
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
                    "GPU pipeline: {} source composed ({} bytes)",
                    stage.label,
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
                            label: Some(stage.label),
                            entries: &layout_entries,
                        })
                    })
                    .collect::<Vec<_>>();
                log::info!("GPU pipeline: {} bind group layouts created", stage.label);
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(stage.label),
                    bind_group_layouts: bind_group_layouts
                        .iter()
                        .map(Some)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    immediate_size: 0,
                });
                log::info!("GPU pipeline: {} pipeline layout created", stage.label);
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(stage.label),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(source)),
                });
                log::info!("GPU pipeline: {} shader module created", stage.label);
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(stage.label),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(stage.entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });
                log::info!("GPU pipeline: created {}", stage.label);
                (
                    stage.id,
                    StagePipeline {
                        pipeline,
                        bind_group_layouts,
                        used_bindings,
                    },
                )
            })
            .collect();
        let pipeline = Self { stages };
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(PbrtError::error(&format!(
                "WebGPU primary-ray pipeline creation failed: {error}"
            )));
        }
        Ok(pipeline)
    }
}

impl StagePipeline {
    pub fn used_bindings(&self) -> &[(u32, u32)] {
        &self.used_bindings
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
