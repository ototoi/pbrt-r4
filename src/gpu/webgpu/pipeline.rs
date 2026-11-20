use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::util::error::PbrtError;

use super::profile::{self, GpuProfiler};
use super::shader;
use super::stage::{ComputeStageId, COMPUTE_STAGES, WAVEFRONT_BIND_GROUP_COUNT};
use super::stages::{canonical_wavefront_bindings, BindingClass, BindingSpec, ResourceId};

pub struct StagePipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub label: &'static str,
    pub profiler: Option<Arc<GpuProfiler>>,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    used_bindings: Vec<(u32, u32)>,
}

pub struct Pipeline {
    stages: HashMap<ComputeStageId, StagePipeline>,
}

impl Pipeline {
    pub fn set_profiler(&mut self, profiler: Arc<GpuProfiler>) {
        for stage in self.stages.values_mut() {
            stage.profiler = Some(profiler.clone());
        }
    }

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
        nanovdb_enabled: bool,
    ) -> Result<Self, PbrtError> {
        Self::new_with_geometry_page_count(
            device,
            texture_image_count,
            texture_sampler_count,
            texture_program_capacity,
            texture_noise_enabled,
            nanovdb_enabled,
            1,
            None,
            None,
        )
    }

    pub fn new_with_geometry_page_count(
        device: &wgpu::Device,
        texture_image_count: u32,
        texture_sampler_count: u32,
        texture_program_capacity: u32,
        texture_noise_enabled: bool,
        nanovdb_enabled: bool,
        geometry_page_count: u32,
        cache_path: Option<&Path>,
        sampler_kind: Option<u32>,
    ) -> Result<Self, PbrtError> {
        if geometry_page_count == 0 {
            return Err(PbrtError::error(
                "WebGPU pipeline requires at least one geometry page.",
            ));
        }
        let cache = cache_path
            .filter(|_| device.features().contains(wgpu::Features::PIPELINE_CACHE))
            .map(|path| {
                let loaded = super::pipeline_cache::load(path);
                unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("pbrt-r4 pipeline cache"),
                        data: loaded.as_deref(),
                        fallback: true,
                    })
                }
            });
        let canonical_bindings = canonical_wavefront_bindings();
        let stages: Result<HashMap<ComputeStageId, StagePipeline>, PbrtError> = COMPUTE_STAGES
            .par_iter()
            .map(|stage| {
                let t = Instant::now();
                log::info!("GPU pipeline: creating {}", stage.label);
                let source = shader::specialize_sampler_kind(
                    shader::specialize_geometry_page_bindings(
                        shader::specialize_source(
                            shader::compose_source_with_features(
                                stage.source,
                                texture_noise_enabled,
                                nanovdb_enabled,
                            ),
                            texture_image_count,
                            texture_sampler_count,
                            texture_program_capacity,
                        ),
                        geometry_page_count,
                    ),
                    sampler_kind,
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
                let stage_pipeline = StagePipeline::create(
                    device,
                    stage.label,
                    stage.entry_point,
                    source,
                    &canonical_bindings,
                    texture_image_count,
                    texture_sampler_count,
                    geometry_page_count,
                    cache.as_ref(),
                )?;
                profile::record_phase(&format!("pipeline: {}", stage.label), t.elapsed());
                Ok((stage.id, stage_pipeline))
            })
            .collect();
        let stages = stages?;
        if let Some(cache) = cache {
            match cache.get_data() {
                Some(data) => {
                    if let Err(error) = super::pipeline_cache::save(cache_path.unwrap(), &data) {
                        log::warn!("Could not save WebGPU pipeline cache: {error}");
                    }
                }
                None => log::warn!("WebGPU pipeline cache returned no data to save."),
            }
        }
        Ok(Self { stages })
    }
}

impl StagePipeline {
    /// Validation errors are scoped to the calling thread, so this can run on any worker.
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        device: &wgpu::Device,
        label: &'static str,
        entry_point: &str,
        source: String,
        canonical_bindings: &[BindingSpec],
        texture_image_count: u32,
        texture_sampler_count: u32,
        geometry_page_count: u32,
        cache: Option<&wgpu::PipelineCache>,
    ) -> Result<Self, PbrtError> {
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bindings = shader::resolve_resource_bindings(canonical_bindings, entry_point, &source)?;
        let used_bindings = bindings
            .iter()
            .map(|binding| (binding.group, binding.binding))
            .collect();
        let bind_group_layouts = (0..WAVEFRONT_BIND_GROUP_COUNT)
            .map(|group| {
                let layout_entries = bindings
                    .iter()
                    .filter(|binding| binding.group == group)
                    .map(|binding| {
                        layout_entry(
                            binding,
                            texture_image_count,
                            texture_sampler_count,
                            geometry_page_count,
                        )
                    })
                    .collect::<Vec<_>>();
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries: &layout_entries,
                })
            })
            .collect::<Vec<_>>();
        log::info!("GPU pipeline: {} bind group layouts created", label);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: bind_group_layouts
                .iter()
                .map(Some)
                .collect::<Vec<_>>()
                .as_slice(),
            immediate_size: 0,
        });
        log::info!("GPU pipeline: {} pipeline layout created", label);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(source)),
        });
        log::info!("GPU pipeline: {} shader module created", label);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(PbrtError::error(&format!(
                "WebGPU {label} pipeline creation failed: {error}"
            )));
        }
        log::info!("GPU pipeline: created {label}");
        Ok(Self {
            pipeline,
            label,
            profiler: None,
            bind_group_layouts,
            used_bindings,
        })
    }

    pub fn used_bindings(&self) -> &[(u32, u32)] {
        &self.used_bindings
    }
}

fn layout_entry(
    binding: &BindingSpec,
    texture_image_count: u32,
    texture_sampler_count: u32,
    geometry_page_count: u32,
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
        ) || (geometry_page_count > 1
            && matches!(binding.resource, ResourceId::Vertex | ResourceId::Index))
        {
            let count = match binding.resource {
                ResourceId::TextureImageArray => texture_image_count,
                ResourceId::TextureSamplerArray => texture_sampler_count,
                ResourceId::Vertex | ResourceId::Index => geometry_page_count,
                _ => unreachable!("only binding arrays have a binding count"),
            };
            std::num::NonZeroU32::new(count.max(1))
        } else {
            None
        },
    }
}
