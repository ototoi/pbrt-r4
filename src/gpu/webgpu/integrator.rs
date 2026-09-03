use std::sync::{Arc, RwLock};

use bytemuck::bytes_of;
use wgpu::util::DeviceExt;

use crate::displays::Display;
use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

use super::abi::{CameraUniform, WORKGROUP_SIZE};
use super::context::Context;
use super::film::Film;
use super::pipeline::Pipeline;
use super::queue::Queues;
use super::scene::Scene;

pub struct WavefrontPathIntegrator {
    context: Context,
    scene: Scene,
    _camera_buffer: wgpu::Buffer,
    _queues: Queues,
    film: Film,
    pipeline: Pipeline,
    bind_group: wgpu::BindGroup,
    rendered: bool,
}

impl WavefrontPathIntegrator {
    pub fn create(flat_scene: flat::Scene) -> Result<Self, PbrtError> {
        let context = Context::new()?;
        let device = &context.device;
        let queue = &context.queue;
        let scene = Scene::from_flat(device, queue, flat_scene)?;
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 camera UBO"),
            contents: bytes_of(&scene.camera),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pixel_count = u64::from(scene.camera.width) * u64::from(scene.camera.height);
        let queues = Queues::new(device, pixel_count);
        let film = Film::new(device, [scene.camera.width, scene.camera.height])?;
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
                    resource: wgpu::BindingResource::AccelerationStructure(
                        &scene.acceleration.tlas,
                    ),
                },
                buffer_entry(2, &scene.vertex_buffer),
                buffer_entry(3, &scene.index_buffer),
                buffer_entry(4, &scene.geometry_buffer),
                buffer_entry(5, &scene.instance_buffer),
                buffer_entry(6, &scene.material_buffer),
                buffer_entry(7, &queues.rays),
                buffer_entry(8, &queues.hits),
                buffer_entry(9, &film.framebuffer),
            ],
        });
        Ok(Self {
            context,
            scene,
            _camera_buffer: camera_buffer,
            _queues: queues,
            film,
            pipeline,
            bind_group,
            rendered: false,
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
        let workgroups_x = self.camera().width.div_ceil(WORKGROUP_SIZE);
        let workgroups_y = self.camera().height.div_ceil(WORKGROUP_SIZE);
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 primary-ray command encoder"),
                });
        self.film.clear(&mut encoder);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pbrt-r4 generate primary rays pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.generate_primary_rays);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pbrt-r4 intersect primary rays pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.intersect_primary_rays);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pbrt-r4 shade normal pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.shade_normal);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        self.film.copy_to_readback(&mut encoder);
        self.context.queue.submit(Some(encoder.finish()));
        self.context.wait()?;
        self.film.readback(&self.context.device)?;
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

    fn camera(&self) -> &CameraUniform {
        &self.scene.camera
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
