use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use bytemuck::bytes_of;
use wgpu::util::DeviceExt;

use crate::displays::Display;
use crate::gpu::ir::flat;
use crate::util::error::PbrtError;
use crate::util::misc::ProgressReporter;

use super::abi::WORKGROUP_SIZE;
use super::context::Context;
use super::film::Film;
use super::material::MaterialKind;
use super::pipeline::{Pipeline, StagePipeline};
use super::queue::Queues;
use super::scene::Scene;
use super::stages::{canonical_wavefront_bindings, ResourceId};

const DEFAULT_DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(500);

const DEPLOYED_STAGE_SOURCES: &[&str] = &[
    include_str!("shaders/prepare_sample.wgsl"),
    include_str!("shaders/generate_primary_rays.wgsl"),
    include_str!("shaders/reset_shadow_queue.wgsl"),
    include_str!("shaders/reset_classification_queues.wgsl"),
    include_str!("shaders/intersect_primary_rays.wgsl"),
    include_str!("shaders/handle_escaped.wgsl"),
    include_str!("shaders/shade_surface.wgsl"),
    include_str!("shaders/handle_emissive.wgsl"),
    include_str!("shaders/evaluate_materials.wgsl"),
    include_str!("shaders/intersect_shadow.wgsl"),
    include_str!("shaders/sample_diffuse_bounce.wgsl"),
    include_str!("shaders/sample_dielectric_bounce.wgsl"),
    include_str!("shaders/sample_conductor_bounce.wgsl"),
    include_str!("shaders/sample_thin_dielectric_bounce.wgsl"),
    include_str!("shaders/sample_composite_bounce.wgsl"),
    include_str!("shaders/swap_ray_queues.wgsl"),
    include_str!("shaders/reset_next_ray_queue.wgsl"),
    include_str!("shaders/accumulate_sample.wgsl"),
];

pub struct WavefrontPathIntegrator {
    context: Context,
    scene: Scene,
    camera_buffer: wgpu::Buffer,
    viewport_buffer: wgpu::Buffer,
    material_table_buffer: wgpu::Buffer,
    light_table_buffer: wgpu::Buffer,
    queues: Queues,
    film: Film,
    pipeline: Pipeline,
    bind_groups: HashMap<&'static str, [wgpu::BindGroup; 2]>,
    rendered: bool,
    show_progress: bool,
}

impl WavefrontPathIntegrator {
    pub fn create(flat_scene: flat::Scene) -> Result<Self, PbrtError> {
        Self::create_with_progress(flat_scene, false)
    }

    pub fn create_with_progress(
        flat_scene: flat::Scene,
        show_progress: bool,
    ) -> Result<Self, PbrtError> {
        let attributes_eval_stride = u64::from(flat::max_attributes_eval_work_items_per_surface(
            &flat_scene,
        )?);
        let canonical_bindings = canonical_wavefront_bindings();
        let mut required_limits = super::shader::required_limits_for_sources(
            &canonical_bindings,
            DEPLOYED_STAGE_SOURCES,
        )?;
        // Every deployed pipeline has a second group reserved for texture
        // binding arrays, even when an individual stage does not sample one.
        required_limits.bind_groups = required_limits.bind_groups.max(2);
        let texture_image_count = u32::try_from(flat_scene.texture_nodes.len())
            .map_err(|_| PbrtError::error("Texture node count exceeds u32 range."))?;
        let texture_sampler_count = texture_image_count;
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
        log::info!("GPU create: WebGPU scene resources ready");
        if let Some(kind) = debug_material {
            scene.replace_material_kind(queue, kind);
            scene.film.mode = 1;
        }
        scene.material_table.attributes_eval_stride = u32::try_from(attributes_eval_stride)
            .map_err(|_| PbrtError::error("GPU attributes eval stride exceeds u32 range."))?;
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
        let pixel_count = u64::from(scene.viewport.width) * u64::from(scene.viewport.height);
        let queues = Queues::new(device, pixel_count, attributes_eval_stride)?;
        log::info!("GPU create: queues and film resources ready");
        let film = Film::new(
            device,
            [scene.viewport.width, scene.viewport.height],
            scene.film_output_matrix,
            scene.film_scale,
            scene.film.mode != 0,
        )?;
        let pipeline = Pipeline::new(
            device,
            scene.texture_images.len() as u32,
            scene.texture_samplers.len() as u32,
        )?;
        log::info!("GPU create: compute pipelines ready");
        let texture_image_views: Vec<&wgpu::TextureView> =
            scene.texture_image_views.iter().collect();
        let texture_samplers: Vec<&wgpu::Sampler> = scene.texture_samplers.iter().collect();
        let make_entry = |binding: super::stages::BindingSpec| wgpu::BindGroupEntry {
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
                ResourceId::HitAreaRayQueue => queues.hit_area_ray_indices.as_entire_binding(),
                ResourceId::EscapedRayQueue => queues.escaped_ray_indices.as_entire_binding(),
                ResourceId::MaterialTable => material_table_buffer.as_entire_binding(),
                ResourceId::LightSamplingParams => light_table_buffer.as_entire_binding(),
                ResourceId::MaterialRecord => scene.material_buffer.as_entire_binding(),
                ResourceId::AttributeRef => scene.attribute_ref_buffer.as_entire_binding(),
                ResourceId::ScalarAttribute => scene.scalar_attribute_buffer.as_entire_binding(),
                ResourceId::SpectrumAttribute => {
                    scene.spectrum_attribute_buffer.as_entire_binding()
                }
                ResourceId::TextureNode => scene.texture_node_buffer.as_entire_binding(),
                ResourceId::TextureChild => scene.texture_child_buffer.as_entire_binding(),
                ResourceId::RgbSpectrumTable => scene.rgb_spectrum_table_buffer.as_entire_binding(),
                ResourceId::TextureImageArray => {
                    wgpu::BindingResource::TextureViewArray(&texture_image_views)
                }
                ResourceId::TextureSamplerArray => {
                    wgpu::BindingResource::SamplerArray(&texture_samplers)
                }
                ResourceId::LightRecord => scene.light_record_buffer.as_entire_binding(),
                ResourceId::LightSamplingModel => {
                    scene.light_sampling_model_buffer.as_entire_binding()
                }
                ResourceId::LightPosition => scene.light_position_buffer.as_entire_binding(),
                ResourceId::TriangleDistribution => scene.distribution_buffer.as_entire_binding(),
                ResourceId::LightBvhHeader => scene.light_bvh_header_buffer.as_entire_binding(),
                ResourceId::LightBvhNode => scene.light_bvh_node_buffer.as_entire_binding(),
                ResourceId::LightLeaf => scene.light_leaf_buffer.as_entire_binding(),
                resource => {
                    panic!("resource {resource:?} is not part of canonical wavefront layout")
                }
            },
        };
        let make_bind_group = |name: &'static str,
                               stage: &StagePipeline,
                               stage_source: &'static str|
         -> [wgpu::BindGroup; 2] {
            let source = super::shader::compose_source(stage_source);
            let used_bindings = super::shader::resource_bindings(&source);
            std::array::from_fn(|group| {
                let entries = canonical_wavefront_bindings()
                    .into_iter()
                    .filter(|binding| {
                        binding.group == group as u32
                            && used_bindings.contains(&(binding.group, binding.binding))
                    })
                    .map(make_entry)
                    .collect::<Vec<_>>();
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(name),
                    layout: &stage.bind_group_layouts[group],
                    entries: &entries,
                })
            })
        };
        let bind_groups: HashMap<&'static str, [wgpu::BindGroup; 2]> = [
            (
                "prepare_sample",
                &pipeline.prepare_sample,
                include_str!("shaders/prepare_sample.wgsl"),
            ),
            (
                "generate_primary_rays",
                &pipeline.generate_primary_rays,
                include_str!("shaders/generate_primary_rays.wgsl"),
            ),
            (
                "reset_shadow_queue",
                &pipeline.reset_shadow_queue,
                include_str!("shaders/reset_shadow_queue.wgsl"),
            ),
            (
                "reset_classification_queues",
                &pipeline.reset_classification_queues,
                include_str!("shaders/reset_classification_queues.wgsl"),
            ),
            (
                "intersect_primary_rays",
                &pipeline.intersect_primary_rays,
                include_str!("shaders/intersect_primary_rays.wgsl"),
            ),
            (
                "handle_escaped",
                &pipeline.handle_escaped,
                include_str!("shaders/handle_escaped.wgsl"),
            ),
            (
                "shade_surface",
                &pipeline.shade_surface,
                include_str!("shaders/shade_surface.wgsl"),
            ),
            (
                "handle_emissive",
                &pipeline.handle_emissive,
                include_str!("shaders/handle_emissive.wgsl"),
            ),
            (
                "evaluate_materials",
                &pipeline.evaluate_materials,
                include_str!("shaders/evaluate_materials.wgsl"),
            ),
            (
                "intersect_shadow",
                &pipeline.intersect_shadow,
                include_str!("shaders/intersect_shadow.wgsl"),
            ),
            (
                "sample_diffuse_bounce",
                &pipeline.sample_diffuse_bounce,
                include_str!("shaders/sample_diffuse_bounce.wgsl"),
            ),
            (
                "sample_dielectric_bounce",
                &pipeline.sample_dielectric_bounce,
                include_str!("shaders/sample_dielectric_bounce.wgsl"),
            ),
            (
                "sample_conductor_bounce",
                &pipeline.sample_conductor_bounce,
                include_str!("shaders/sample_conductor_bounce.wgsl"),
            ),
            (
                "sample_thin_dielectric_bounce",
                &pipeline.sample_thin_dielectric_bounce,
                include_str!("shaders/sample_thin_dielectric_bounce.wgsl"),
            ),
            (
                "sample_composite_bounce",
                &pipeline.sample_composite_bounce,
                include_str!("shaders/sample_composite_bounce.wgsl"),
            ),
            (
                "swap_ray_queues",
                &pipeline.swap_ray_queues,
                include_str!("shaders/swap_ray_queues.wgsl"),
            ),
            (
                "reset_next_ray_queue",
                &pipeline.reset_next_ray_queue,
                include_str!("shaders/reset_next_ray_queue.wgsl"),
            ),
            (
                "accumulate_sample",
                &pipeline.accumulate_sample,
                include_str!("shaders/accumulate_sample.wgsl"),
            ),
        ]
        .into_iter()
        .map(|(name, stage, source)| (name, make_bind_group(name, stage, source)))
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
            pipeline,
            bind_groups,
            rendered: false,
            show_progress,
        })
    }

    fn bind_groups(&self, name: &'static str) -> &[wgpu::BindGroup; 2] {
        self.bind_groups
            .get(name)
            .expect("stage bind group is registered")
    }

    pub fn add_display(&mut self, display: &Arc<RwLock<dyn Display>>) {
        self.film.add_display(display);
    }

    pub fn render(&mut self) -> Result<(), PbrtError> {
        if self.rendered {
            return Err(PbrtError::error(
                "The initial WebGPU primary-ray integrator can only render once.",
            ));
        }
        if let Err(error) = self.film.start() {
            log::warn!("WebGPU Film display start failed: {error}");
        }
        let workgroups_x = self.scene.viewport.width.div_ceil(WORKGROUP_SIZE);
        let workgroups_y = self.scene.viewport.height.div_ceil(WORKGROUP_SIZE);
        let samples_per_pixel = self.scene.render_settings.samples_per_pixel;
        let mut reporter = self.show_progress.then(|| {
            ProgressReporter::new(samples_per_pixel as usize, &self.scene.output.filename)
        });
        let mut last_display_update = Instant::now();
        log::info!(
            "GPU render: starting samples={samples_per_pixel} depth={}",
            self.scene.render_settings.max_depth
        );
        for sample_index in 0..samples_per_pixel {
            log::info!(
                "GPU render: sample {}/{}",
                sample_index + 1,
                samples_per_pixel
            );
            self.scene.viewport.sample_index = sample_index;
            self.context.queue.write_buffer(
                &self.viewport_buffer,
                0,
                bytes_of(&self.scene.viewport),
            );
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pbrt-r4 diffuse command encoder"),
                    });
            if sample_index == 0 {
                self.film.clear(&mut encoder);
            }
            dispatch(
                &mut encoder,
                &self.pipeline.prepare_sample.pipeline,
                self.bind_groups("prepare_sample"),
                workgroups_x,
                workgroups_y,
            );
            dispatch(
                &mut encoder,
                &self.pipeline.generate_primary_rays.pipeline,
                self.bind_groups("generate_primary_rays"),
                workgroups_x,
                workgroups_y,
            );
            for depth in 0..=self.scene.render_settings.max_depth {
                if depth != 0 {
                    dispatch(
                        &mut encoder,
                        &self.pipeline.reset_shadow_queue.pipeline,
                        self.bind_groups("reset_shadow_queue"),
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.reset_classification_queues.pipeline,
                        self.bind_groups("reset_classification_queues"),
                        workgroups_x,
                        workgroups_y,
                    );
                }
                dispatch(
                    &mut encoder,
                    &self.pipeline.intersect_primary_rays.pipeline,
                    self.bind_groups("intersect_primary_rays"),
                    workgroups_x,
                    workgroups_y,
                );
                dispatch(
                    &mut encoder,
                    &self.pipeline.handle_escaped.pipeline,
                    self.bind_groups("handle_escaped"),
                    workgroups_x,
                    workgroups_y,
                );
                dispatch(
                    &mut encoder,
                    &self.pipeline.shade_surface.pipeline,
                    self.bind_groups("shade_surface"),
                    workgroups_x,
                    workgroups_y,
                );
                dispatch(
                    &mut encoder,
                    &self.pipeline.handle_emissive.pipeline,
                    self.bind_groups("handle_emissive"),
                    workgroups_x,
                    workgroups_y,
                );
                dispatch(
                    &mut encoder,
                    &self.pipeline.evaluate_materials.pipeline,
                    self.bind_groups("evaluate_materials"),
                    workgroups_x,
                    workgroups_y,
                );
                if depth < self.scene.render_settings.max_depth {
                    dispatch(
                        &mut encoder,
                        &self.pipeline.intersect_shadow.pipeline,
                        self.bind_groups("intersect_shadow"),
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.sample_diffuse_bounce.pipeline,
                        self.bind_groups("sample_diffuse_bounce"),
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.sample_dielectric_bounce.pipeline,
                        self.bind_groups("sample_dielectric_bounce"),
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.sample_conductor_bounce.pipeline,
                        self.bind_groups("sample_conductor_bounce"),
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.sample_thin_dielectric_bounce.pipeline,
                        self.bind_groups("sample_thin_dielectric_bounce"),
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.sample_composite_bounce.pipeline,
                        self.bind_groups("sample_composite_bounce"),
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.swap_ray_queues.pipeline,
                        self.bind_groups("swap_ray_queues"),
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.reset_next_ray_queue.pipeline,
                        self.bind_groups("reset_next_ray_queue"),
                        workgroups_x,
                        workgroups_y,
                    );
                }
            }
            dispatch(
                &mut encoder,
                &self.pipeline.accumulate_sample.pipeline,
                self.bind_groups("accumulate_sample"),
                workgroups_x,
                workgroups_y,
            );
            self.context.queue.submit(Some(encoder.finish()));
            log::info!("GPU render: submitted sample {sample_index}; waiting for film completion");
            self.film.complete_sample()?;
            log::info!("GPU render: sample {sample_index} complete");
            let completed_samples = self.film.completed_samples();
            if !self.film.has_no_display()
                && (last_display_update.elapsed() >= DEFAULT_DISPLAY_UPDATE_INTERVAL
                    || completed_samples == samples_per_pixel)
            {
                let mut display_encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("pbrt-r4 WebGPU display readback encoder"),
                        });
                self.film.copy_to_readback(&mut display_encoder);
                self.queues.copy_state_to_readback(&mut display_encoder);
                self.context.queue.submit(Some(display_encoder.finish()));
                self.context.wait()?;
                if self.queues.read_error(&self.context.device)? {
                    return Err(PbrtError::error(
                        "WebGPU wavefront rendering reported an error.",
                    ));
                }
                self.film.readback(&self.context.device)?;
                if let Err(error) = self.film.update_display() {
                    log::warn!("WebGPU Film display update failed: {error}");
                }
                last_display_update = Instant::now();
            }
            if let Some(reporter) = reporter.as_mut() {
                reporter.update(1);
            }
        }
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 diffuse readback encoder"),
                });
        self.film.copy_to_readback(&mut encoder);
        self.queues.copy_state_to_readback(&mut encoder);
        self.context.queue.submit(Some(encoder.finish()));
        self.context.wait()?;
        if self.queues.read_error(&self.context.device)? {
            return Err(PbrtError::error(
                "WebGPU wavefront rendering reported an error.",
            ));
        }
        self.film.readback(&self.context.device)?;
        if let Some(reporter) = reporter.as_mut() {
            reporter.done();
        }
        if let Err(error) = self.film.update_display() {
            log::warn!("WebGPU Film display update failed: {error}");
        }
        if let Err(error) = self.film.end() {
            log::warn!("WebGPU Film display end failed: {error}");
        }
        self.film.write_output(&self.scene.output)?;
        self.rendered = true;
        Ok(())
    }

    pub fn replace_material_kind(&mut self, kind: super::material::MaterialKind) {
        self.scene.replace_material_kind(&self.context.queue, kind);
        self.context.queue.write_buffer(
            &self.material_table_buffer,
            0,
            bytes_of(&self.scene.material_table),
        );
    }
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_groups: &[wgpu::BindGroup; 2],
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_groups[0], &[]);
    pass.set_bind_group(1, &bind_groups[1], &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
