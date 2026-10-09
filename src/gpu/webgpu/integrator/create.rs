use std::collections::HashMap;

use bytemuck::bytes_of;
use wgpu::util::DeviceExt;

use crate::gpu::flat;
use crate::gpu::flat::texture::{ProceduralOperation, TextureInstruction};
use crate::util::error::PbrtError;

use super::super::abi::{BSSRDFProbeResult, BSSRDFProbeWorkItem};
use super::super::bssrdf::BSSRDFProbePipeline;
use super::super::context::Context;
use super::super::film::Film;
use super::super::material::MaterialKind;
use super::super::noise::NoiseRuntimeResources;
use super::super::pipeline::Pipeline;
use super::super::queue::Queues;
use super::super::scene::{texture_binding_counts, Scene};
use super::super::shader::required_limits_for_sources;
use super::super::stage::{ComputeStageSpec, COMPUTE_STAGES};
use super::super::stages::{canonical_wavefront_bindings, BindingSpec, ResourceId};
use super::tiles::DEFAULT_GPU_TILE_SIZE;

impl super::WavefrontPathIntegrator {
    pub fn create(flat_scene: flat::Scene) -> Result<Self, PbrtError> {
        Self::create_with_progress(flat_scene, false, None)
    }

    pub fn create_with_progress(
        flat_scene: flat::Scene,
        show_progress: bool,
        tile_size: Option<u32>,
    ) -> Result<Self, PbrtError> {
        let has_interface_only_instances = flat_scene
            .instances
            .iter()
            .any(|instance| instance.material_root == flat::INVALID_INDEX);
        let attributes_eval_stride = u64::from(flat::max_attributes_eval_work_items_per_surface(
            &flat_scene,
        )?);
        let texture_eval_stride =
            u64::from(flat::max_texture_eval_results_per_surface(&flat_scene)?).max(1);
        let canonical_bindings = canonical_wavefront_bindings();
        let mut required_limits = required_limits_for_sources(
            &canonical_bindings,
            &COMPUTE_STAGES
                .iter()
                .map(|stage| stage.source)
                .collect::<Vec<_>>(),
        )?;
        // Every deployed pipeline has a second group reserved for texture
        // binding arrays, even when an individual stage does not sample one.
        required_limits.bind_groups = required_limits.bind_groups.max(2);
        let (mut texture_image_count, texture_sampler_count) = texture_binding_counts(&flat_scene)?;
        texture_image_count = texture_image_count
            .checked_add(
                u32::try_from(flat_scene.measured_bsdfs.atlas_pages.len())
                    .map_err(|_| PbrtError::error("Measured BSDF atlas page count exceeds u32."))?,
            )
            .ok_or_else(|| PbrtError::error("Texture image binding count overflowed."))?;
        let texture_program_capacity = flat_scene
            .texture_library
            .programs
            .iter()
            .map(|program| program.instructions.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let texture_program_capacity = u32::try_from(texture_program_capacity)
            .map_err(|_| PbrtError::error("Texture program capacity exceeds u32."))?;
        let texture_noise_enabled = flat_scene.texture_library.programs.iter().any(|program| {
            program.instructions.iter().any(|instruction| {
                matches!(
                    instruction,
                    TextureInstruction::Procedural {
                        operation: ProceduralOperation::Dots
                            | ProceduralOperation::Fbm
                            | ProceduralOperation::Wrinkled
                            | ProceduralOperation::Windy
                            | ProceduralOperation::Marble,
                        ..
                    }
                )
            })
        });
        let medium_scattering_enabled = flat_scene.media.iter().any(|medium| {
            flat_scene
                .spectrum_attributes
                .get(medium.sigma_s as usize)
                .is_some_and(|spectrum| spectrum.samples.iter().any(|sample| *sample != 0.0))
        });
        log::info!(
            "GPU create: requesting WebGPU context (texture_images={texture_image_count}, texture_samplers={texture_sampler_count})"
        );
        let context = Context::new(
            required_limits,
            texture_image_count.max(1),
            texture_sampler_count.max(1),
        )?;
        log::info!("GPU create: WebGPU context ready");
        let device = &context.device;
        let queue = &context.queue;
        let debug_material = MaterialKind::from_debug_environment()?;
        let mut scene = Scene::from_flat(device, queue, flat_scene)?;
        scene.viewport.medium_scattering_enabled = u32::from(medium_scattering_enabled);
        log::info!("GPU create: WebGPU scene resources ready");
        if let Some(kind) = debug_material {
            scene.replace_material_kind(kind);
            scene.film.mode = 1;
        }
        scene.material_table.attributes_eval_stride = u32::try_from(attributes_eval_stride)
            .map_err(|_| PbrtError::error("GPU attributes eval stride exceeds u32 range."))?;
        scene.material_table.texture_eval_stride = u32::try_from(texture_eval_stride)
            .map_err(|_| PbrtError::error("GPU texture eval stride exceeds u32 range."))?;
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 camera UBO"),
            contents: bytes_of(&scene.camera),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let viewport_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 viewport UBO"),
            contents: bytes_of(&scene.viewport),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let film_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 film UBO"),
            contents: bytemuck::bytes_of(&scene.film),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let material_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material table UBO"),
            contents: bytes_of(&scene.material_table),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let light_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light table UBO"),
            contents: bytes_of(&scene.light_table),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let tile_size = tile_size.unwrap_or(DEFAULT_GPU_TILE_SIZE).max(1);
        let tile_width = tile_size.min(scene.viewport.region_width);
        let tile_height = tile_size.min(scene.viewport.region_height);
        scene.viewport.tile_x = scene.viewport.region_x;
        scene.viewport.tile_y = scene.viewport.region_y;
        scene.viewport.tile_width = tile_width;
        scene.viewport.tile_height = tile_height;
        let tile_pixel_count = u64::from(tile_width) * u64::from(tile_height);
        let queues = Queues::new(
            device,
            tile_pixel_count,
            attributes_eval_stride,
            texture_eval_stride,
        )?;
        let bssrdf_capacity = if scene.material_table.have_subsurface != 0 {
            tile_pixel_count
        } else {
            1
        };
        let work_size = 16 + bssrdf_capacity * std::mem::size_of::<BSSRDFProbeWorkItem>() as u64;
        let result_size = bssrdf_capacity * std::mem::size_of::<BSSRDFProbeResult>() as u64;
        let storage_limit = u64::from(device.limits().max_storage_buffer_binding_size);
        if work_size > storage_limit || result_size > storage_limit {
            return Err(PbrtError::error(
                "BSSRDF queues exceed device storage limits.",
            ));
        }
        let bssrdf_work = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BSSRDF probe work"),
            size: work_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bssrdf_results = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BSSRDF probe results"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bssrdf_probe = if scene.material_table.have_subsurface != 0 {
            let pipeline = BSSRDFProbePipeline::new(device)?;
            let bindings = pipeline.bind_group(
                device,
                &scene,
                &scene.bssrdf_tables,
                &bssrdf_work,
                &bssrdf_results,
                &queues.render_error,
            )?;
            Some((pipeline, bindings))
        } else {
            None
        };
        let noise_resources = NoiseRuntimeResources::new(device, queue);
        log::info!("GPU create: queues and film resources ready");
        let film = Film::new(
            device,
            [scene.viewport.region_width, scene.viewport.region_height],
            [scene.viewport.full_width, scene.viewport.full_height],
            [scene.viewport.region_x, scene.viewport.region_y],
            scene.film_output_matrix,
            scene.film_scale,
            scene.film.mode != 0,
        )?;
        let pipeline = Pipeline::new(
            device,
            scene.texture_images.len() as u32,
            scene.texture_samplers.len() as u32,
            texture_program_capacity,
            texture_noise_enabled,
        )?;
        log::info!("GPU create: compute pipelines ready");
        let texture_image_views: Vec<&wgpu::TextureView> =
            scene.texture_image_views.iter().collect();
        let texture_samplers: Vec<&wgpu::Sampler> = scene.texture_samplers.iter().collect();
        let make_entry = |binding: BindingSpec| wgpu::BindGroupEntry {
            binding: binding.binding,
            resource: match binding.resource {
                ResourceId::CameraParams => camera_buffer.as_entire_binding(),
                ResourceId::SampleParams => viewport_buffer.as_entire_binding(),
                ResourceId::Tlas => {
                    wgpu::BindingResource::AccelerationStructure(&scene.acceleration.tlas)
                }
                ResourceId::Vertex => scene.vertex_buffer.as_entire_binding(),
                ResourceId::Index => scene.index_buffer.as_entire_binding(),
                ResourceId::Geometry => scene.geometry_buffer.as_entire_binding(),
                ResourceId::Instance => scene.instance_buffer.as_entire_binding(),
                ResourceId::Medium => scene.medium_buffer.as_entire_binding(),
                ResourceId::BSSRDFMaterial => scene.bssrdf_material_buffer.as_entire_binding(),
                ResourceId::BSSRDFTable => scene.bssrdf_tables.records.as_entire_binding(),
                ResourceId::BSSRDFValues => scene.bssrdf_tables.values.as_entire_binding(),
                ResourceId::BSSRDFWork => bssrdf_work.as_entire_binding(),
                ResourceId::BSSRDFResults => bssrdf_results.as_entire_binding(),
                ResourceId::ActiveMediumIndices => queues.active_medium_indices.as_entire_binding(),
                ResourceId::NextMediumIndices => queues.next_medium_indices.as_entire_binding(),
                ResourceId::ActiveShadowIndices => queues.active_shadow_indices.as_entire_binding(),
                ResourceId::NextShadowIndices => queues.next_shadow_indices.as_entire_binding(),
                ResourceId::MediumScatterQueue => queues.medium_scatter_indices.as_entire_binding(),
                ResourceId::FilmParams => film_params_buffer.as_entire_binding(),
                ResourceId::Surface => queues.surfaces.as_entire_binding(),
                ResourceId::Film => film.framebuffer.as_entire_binding(),
                ResourceId::QueueCounters => queues.counters.as_entire_binding(),
                ResourceId::RenderError => queues.render_error.as_entire_binding(),
                ResourceId::PixelSampleState => queues.pixel_sample_states.as_entire_binding(),
                ResourceId::CurrentRay => queues.current_rays.as_entire_binding(),
                ResourceId::NextRay => queues.next_rays.as_entire_binding(),
                ResourceId::ShadowQueue => queues.shadow_rays.as_entire_binding(),
                ResourceId::MaterialRayQueue => queues.material_ray_indices.as_entire_binding(),
                ResourceId::AttributesEvalWorkItems => {
                    queues.attributes_eval_work_items.as_entire_binding()
                }
                ResourceId::TextureEvalResult => queues.texture_eval_results.as_entire_binding(),
                ResourceId::HitAreaRayQueue => queues.hit_area_ray_indices.as_entire_binding(),
                ResourceId::EscapedRayQueue => queues.escaped_ray_indices.as_entire_binding(),
                ResourceId::MaterialTable => material_table_buffer.as_entire_binding(),
                ResourceId::LightSamplingParams => light_table_buffer.as_entire_binding(),
                ResourceId::SamplerParams | ResourceId::SamplerTable => scene
                    .sampler
                    .bindings()
                    .resource(binding.resource)
                    .expect("sampler resource binding"),
                ResourceId::MaterialRoot => scene.material_root_buffer.as_entire_binding(),
                ResourceId::MaterialNode => scene.material_node_buffer.as_entire_binding(),
                ResourceId::AttributeRef => scene.attribute_ref_buffer.as_entire_binding(),
                ResourceId::ScalarAttribute => scene.scalar_attribute_buffer.as_entire_binding(),
                ResourceId::SpectrumAttribute => {
                    scene.spectrum_attribute_buffer.as_entire_binding()
                }
                ResourceId::MeasuredBsdf => scene.measured_bsdf_buffer.as_entire_binding(),
                ResourceId::MeasuredTable => scene.measured_table_buffer.as_entire_binding(),
                ResourceId::TextureNode => scene.texture_node_buffer.as_entire_binding(),
                ResourceId::TextureRoot => scene.texture_root_buffer.as_entire_binding(),
                ResourceId::TextureChild => scene.texture_child_buffer.as_entire_binding(),
                ResourceId::RgbSpectrumTable => scene.rgb_spectrum_table_buffer.as_entire_binding(),
                ResourceId::TextureImageArray => {
                    wgpu::BindingResource::TextureViewArray(&texture_image_views)
                }
                ResourceId::TextureSamplerArray => {
                    wgpu::BindingResource::SamplerArray(&texture_samplers)
                }
                ResourceId::NoiseTable => wgpu::BindingResource::TextureView(&noise_resources.view),
                ResourceId::LightRecord => scene.light_record_buffer.as_entire_binding(),
                ResourceId::LightSamplingModel => {
                    scene.light_sampling_model_buffer.as_entire_binding()
                }
                ResourceId::LightPosition => scene.light_position_buffer.as_entire_binding(),
                ResourceId::TriangleDistribution => scene.distribution_buffer.as_entire_binding(),
                ResourceId::PortalInfiniteLight => scene.portal_image_buffer.as_entire_binding(),
                ResourceId::PortalDistribution => {
                    scene.portal_distribution_buffer.as_entire_binding()
                }
                ResourceId::ImageInfiniteSampling => {
                    scene.image_infinite_sampling_buffer.as_entire_binding()
                }
                ResourceId::ImageInfiniteDistribution => {
                    scene.image_infinite_distribution_buffer.as_entire_binding()
                }
                ResourceId::ImageInfiniteRowCdf => {
                    scene.image_infinite_row_cdf_buffer.as_entire_binding()
                }
                ResourceId::DirectLightSample => queues.direct_light_samples.as_entire_binding(),
                ResourceId::DirectEvalQueue => queues.direct_eval_ray_indices.as_entire_binding(),
                ResourceId::ScatterDiffuseQueue => {
                    queues.scatter_diffuse_ray_indices.as_entire_binding()
                }
                ResourceId::ScatterDiffuseTransmissionQueue => queues
                    .scatter_diffuse_transmission_ray_indices
                    .as_entire_binding(),
                ResourceId::ScatterConductorQueue => {
                    queues.scatter_conductor_ray_indices.as_entire_binding()
                }
                ResourceId::ScatterDielectricQueue => {
                    queues.scatter_dielectric_ray_indices.as_entire_binding()
                }
                ResourceId::ScatterThinDielectricQueue => queues
                    .scatter_thin_dielectric_ray_indices
                    .as_entire_binding(),
                ResourceId::ScatterMeasuredQueue => {
                    queues.scatter_measured_ray_indices.as_entire_binding()
                }
                ResourceId::ScatterCoatedQueue => {
                    queues.scatter_coated_ray_indices.as_entire_binding()
                }
                ResourceId::QueueDispatchArgs => queues.queue_dispatch_args.as_entire_binding(),
                ResourceId::LightBvhHeader => scene.light_bvh_header_buffer.as_entire_binding(),
                ResourceId::LightBvhNode => scene.light_bvh_node_buffer.as_entire_binding(),
                ResourceId::LightLeaf => scene.light_leaf_buffer.as_entire_binding(),
                resource => {
                    panic!("resource {resource:?} is not part of canonical wavefront layout")
                }
            },
        };
        let make_bind_group = |stage_spec: &ComputeStageSpec| -> [wgpu::BindGroup; 2] {
            let stage_pipeline = pipeline.stage(stage_spec.id);
            std::array::from_fn(|group| {
                let entries = canonical_bindings
                    .iter()
                    .filter(|binding| {
                        binding.group == group as u32
                            && stage_pipeline
                                .used_bindings()
                                .contains(&(binding.group, binding.binding))
                    })
                    .copied()
                    .map(make_entry)
                    .collect::<Vec<_>>();
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(stage_spec.label),
                    layout: &stage_pipeline.bind_group_layouts[group],
                    entries: &entries,
                })
            })
        };
        let bind_groups = COMPUTE_STAGES
            .iter()
            .map(|stage| (stage.id, make_bind_group(stage)))
            .collect::<HashMap<_, _>>();
        log::info!("GPU create: bind groups ready");
        Ok(Self {
            context,
            scene,
            camera_buffer,
            viewport_buffer,
            material_table_buffer,
            light_table_buffer,
            queues,
            film,
            noise_resources,
            pipeline,
            bind_groups,
            rendered: false,
            show_progress,
            tile_width,
            tile_height,
            has_interface_only_instances,
            bssrdf_work,
            _bssrdf_results: bssrdf_results,
            bssrdf_probe,
        })
    }
}
