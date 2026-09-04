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
use super::pipeline::Pipeline;
use super::queue::Queues;
use super::scene::Scene;

const DEFAULT_DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(500);

pub struct WavefrontPathIntegrator {
    context: Context,
    scene: Scene,
    camera_buffer: wgpu::Buffer,
    viewport_buffer: wgpu::Buffer,
    queues: Queues,
    film: Film,
    pipeline: Pipeline,
    bind_group: wgpu::BindGroup,
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
        let context = Context::new()?;
        let device = &context.device;
        let queue = &context.queue;
        let mut scene = Scene::from_flat(device, queue, flat_scene)?;
        scene.replace_material_kind(queue, MaterialKind::from_debug_environment()?);
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
        let pixel_count = u64::from(scene.viewport.width) * u64::from(scene.viewport.height);
        let queues = Queues::new(device, pixel_count, scene.render_settings.max_depth)?;
        let film = Film::new(device, [scene.viewport.width, scene.viewport.height])?;
        let pipeline = Pipeline::new(device)?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbrt-r4 primary-ray bind group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: viewport_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::AccelerationStructure(
                        &scene.acceleration.tlas,
                    ),
                },
                buffer_entry(3, &scene.vertex_buffer),
                buffer_entry(4, &scene.index_buffer),
                buffer_entry(5, &scene.geometry_buffer),
                buffer_entry(6, &scene.instance_buffer),
                buffer_entry(7, &scene.material_buffer),
                buffer_entry(8, &queues.surfaces),
                buffer_entry(9, &film.framebuffer),
                buffer_entry(10, &queues.wavefront),
            ],
        });
        Ok(Self {
            context,
            scene,
            camera_buffer,
            viewport_buffer,
            queues,
            film,
            pipeline,
            bind_group,
            rendered: false,
            show_progress,
        })
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
        for sample_index in 0..samples_per_pixel {
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
                &self.pipeline.prepare_sample,
                &self.bind_group,
                workgroups_x,
                workgroups_y,
            );
            dispatch(
                &mut encoder,
                &self.pipeline.generate_primary_rays,
                &self.bind_group,
                workgroups_x,
                workgroups_y,
            );
            for depth in 0..=self.scene.render_settings.max_depth {
                if depth != 0 {
                    dispatch(
                        &mut encoder,
                        &self.pipeline.reset_shadow_queue,
                        &self.bind_group,
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.reset_classification_queues,
                        &self.bind_group,
                        workgroups_x,
                        workgroups_y,
                    );
                }
                dispatch(
                    &mut encoder,
                    &self.pipeline.intersect_primary_rays,
                    &self.bind_group,
                    workgroups_x,
                    workgroups_y,
                );
                dispatch(
                    &mut encoder,
                    &self.pipeline.handle_escaped,
                    &self.bind_group,
                    workgroups_x,
                    workgroups_y,
                );
                dispatch(
                    &mut encoder,
                    &self.pipeline.shade_surface,
                    &self.bind_group,
                    workgroups_x,
                    workgroups_y,
                );
                dispatch(
                    &mut encoder,
                    &self.pipeline.handle_emissive,
                    &self.bind_group,
                    workgroups_x,
                    workgroups_y,
                );
                dispatch(
                    &mut encoder,
                    &self.pipeline.evaluate_materials,
                    &self.bind_group,
                    workgroups_x,
                    workgroups_y,
                );
                if depth < self.scene.render_settings.max_depth {
                    dispatch(
                        &mut encoder,
                        &self.pipeline.intersect_shadow,
                        &self.bind_group,
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.finish_shadow,
                        &self.bind_group,
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.sample_diffuse_bounce,
                        &self.bind_group,
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.swap_ray_queues,
                        &self.bind_group,
                        workgroups_x,
                        workgroups_y,
                    );
                    dispatch(
                        &mut encoder,
                        &self.pipeline.reset_next_ray_queue,
                        &self.bind_group,
                        workgroups_x,
                        workgroups_y,
                    );
                }
            }
            dispatch(
                &mut encoder,
                &self.pipeline.accumulate_sample,
                &self.bind_group,
                workgroups_x,
                workgroups_y,
            );
            self.context.queue.submit(Some(encoder.finish()));
            self.film.complete_sample()?;
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
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
