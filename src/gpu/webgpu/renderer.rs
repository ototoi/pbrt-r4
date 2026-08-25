use super::super::compiler::GpuCompiledScene;
use super::super::ir::{GpuMatrix4x4, GpuRenderOutput, GpuRenderRequest, GpuSceneView};
use super::device::{AccelerationMode, DeviceContext, PrepareOptions};
use super::error::BackendError;
use super::geometry::{HardwareAcceleration, ScenePlan};
use super::shader::{build_shader_set, ShaderStageId};
use super::software::SoftwareAcceleration;
use super::wavefront::WavefrontLayout;
use std::time::{Duration, Instant};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

const WAVEFRONT_READBACK_SPP_INTERVAL: u32 = 8;
const WAVEFRONT_READBACK_TIME_INTERVAL: Duration = Duration::from_secs(2);

enum SceneResources {
    Hardware(HardwareAcceleration),
    Software(SoftwareAcceleration),
}

pub struct ExecutableScene {
    scene: GpuCompiledScene,
    resources: SceneResources,
    supports_wavefront_min: bool,
}

impl ExecutableScene {
    pub fn scene(&self) -> GpuSceneView<'_> {
        self.scene.view()
    }
}

struct Pipeline {
    stages: Vec<(ShaderStageId, wgpu::ComputePipeline)>,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Pipeline {
    fn stage(&self, id: ShaderStageId) -> &wgpu::ComputePipeline {
        &self
            .stages
            .iter()
            .find(|(stage_id, _)| *stage_id == id)
            .unwrap_or_else(|| panic!("WebGPU pipeline does not contain stage {id:?}"))
            .1
    }
}

pub struct Renderer {
    device_context: DeviceContext,
    pipeline: Pipeline,
}

impl Renderer {
    pub fn new(options: &PrepareOptions) -> Result<Self, BackendError> {
        let device_context = DeviceContext::create(options)?;
        let pipeline = create_pipeline(&device_context, options.acceleration_mode);
        Ok(Self {
            device_context,
            pipeline,
        })
    }

    pub fn prepare(&mut self, scene: &GpuCompiledScene) -> Result<ExecutableScene, BackendError> {
        let plan = ScenePlan::from_scene(scene.view())?;
        let supports_wavefront_min = plan.supports_wavefront_min(scene.view());
        let resources = match self.device_context.acceleration_mode {
            AccelerationMode::HardwareRayQuery => {
                SceneResources::Hardware(HardwareAcceleration::create(&self.device_context, &plan)?)
            }
            AccelerationMode::SoftwareBvh => {
                SceneResources::Software(SoftwareAcceleration::create(&self.device_context, &plan)?)
            }
        };
        Ok(ExecutableScene {
            scene: scene.clone(),
            resources,
            supports_wavefront_min,
        })
    }

    pub fn render(
        &mut self,
        scene: &ExecutableScene,
        request: &GpuRenderRequest,
    ) -> Result<GpuRenderOutput, BackendError> {
        let scene_view = scene.scene();
        let request = GpuRenderRequest::new(
            scene_view.render,
            request.sample_start,
            request.sample_count,
        )
        .map_err(BackendError::InvalidRenderRequest)?;
        let pixel_bounds = scene_view.render.film.pixel_bounds;
        let width = pixel_bounds.max[0]
            .checked_sub(pixel_bounds.min[0])
            .ok_or(BackendError::Readback("invalid pixel bounds".to_string()))?;
        let height = pixel_bounds.max[1]
            .checked_sub(pixel_bounds.min[1])
            .ok_or(BackendError::Readback("invalid pixel bounds".to_string()))?;
        if width == 0 || height == 0 {
            return Err(BackendError::Readback(
                "pixel bounds must contain at least one pixel".to_string(),
            ));
        }
        let pixel_count = usize::try_from(width)
            .ok()
            .and_then(|width| usize::try_from(height).ok()?.checked_mul(width))
            .ok_or(BackendError::Readback("pixel count overflow".to_string()))?;
        let output_size = wgpu::BufferAddress::try_from(
            pixel_count
                .checked_mul(std::mem::size_of::<[f32; 4]>())
                .ok_or(BackendError::Readback("output size overflow".to_string()))?,
        )
        .map_err(|_| BackendError::Readback("output size overflow".to_string()))?;
        // The hardware wavefront index arena shares the film storage binding.
        // One vec4 is reserved per index to keep concurrent writes independent.
        // Ray A/B, classification queues, and the shadow queue each have a
        // disjoint region in this index arena.
        let queue_index_size = output_size
            .checked_mul(6)
            .ok_or_else(|| BackendError::Readback("ray queue index size overflow".to_string()))?;
        let output_buffer_size = output_size
            .checked_add(queue_index_size)
            .ok_or_else(|| BackendError::Readback("film buffer size overflow".to_string()))?;
        let wavefront_layout = WavefrontLayout::for_pixel_count(
            u32::try_from(pixel_count)
                .map_err(|_| BackendError::Readback("pixel count exceeds u32".to_string()))?,
        )
        .ok_or_else(|| BackendError::Readback("wavefront arena size overflow".to_string()))?;

        let (bvh_primitive_offset, bvh_node_offset) = match &scene.resources {
            SceneResources::Hardware(_) => (0, 0),
            SceneResources::Software(acceleration) => (
                acceleration.bvh_primitive_offset,
                acceleration.bvh_node_offset,
            ),
        };
        let use_wavefront_min =
            matches!(&scene.resources, SceneResources::Hardware(_)) && scene.supports_wavefront_min;
        let uniform_buffer = self
            .device_context
            .device
            .create_buffer_init(&BufferInitDescriptor {
                label: Some("pbrt-r4 WebGPU camera uniforms"),
                contents: &camera_uniform_bytes(
                    scene_view,
                    request,
                    bvh_primitive_offset,
                    bvh_node_offset,
                ),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let output_buffer = self
            .device_context
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 WebGPU film output"),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        let readback_buffer = self
            .device_context
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 WebGPU film readback"),
                size: output_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
        let control_readback_buffer =
            self.device_context
                .device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pbrt-r4 WebGPU wavefront control readback"),
                    size: u64::from(super::wavefront::WAVEFRONT_CONTROL_SIZE),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
        let wavefront_buffer = self
            .device_context
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 WebGPU wavefront arena"),
                size: u64::from(wavefront_layout.byte_len),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        let bind_group =
            match &scene.resources {
                SceneResources::Hardware(acceleration) => self
                    .device_context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("pbrt-r4 WebGPU hardware bind group"),
                        layout: &self.pipeline.bind_group_layout,
                        entries: &[
                            buffer_entry(0, uniform_buffer.as_entire_binding()),
                            buffer_entry(1, output_buffer.as_entire_binding()),
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::AccelerationStructure(
                                    &acceleration.tlas,
                                ),
                            },
                            buffer_entry(3, acceleration.vertex_buffer.as_entire_binding()),
                            buffer_entry(4, acceleration.index_buffer.as_entire_binding()),
                            buffer_entry(5, acceleration.primitive_buffer.as_entire_binding()),
                            buffer_entry(6, acceleration.transform_buffer.as_entire_binding()),
                            buffer_entry(7, acceleration.material_buffer.as_entire_binding()),
                            buffer_entry(8, acceleration.light_buffer.as_entire_binding()),
                            buffer_entry(9, acceleration.texture_buffer.as_entire_binding()),
                            buffer_entry(10, wavefront_buffer.as_entire_binding()),
                        ],
                    }),
                SceneResources::Software(acceleration) => self
                    .device_context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("pbrt-r4 WebGPU software BVH bind group"),
                        layout: &self.pipeline.bind_group_layout,
                        entries: &[
                            buffer_entry(0, uniform_buffer.as_entire_binding()),
                            buffer_entry(1, output_buffer.as_entire_binding()),
                            buffer_entry(3, acceleration.vertex_buffer.as_entire_binding()),
                            buffer_entry(4, acceleration.index_buffer.as_entire_binding()),
                            buffer_entry(5, acceleration.primitive_buffer.as_entire_binding()),
                            buffer_entry(6, acceleration.transform_buffer.as_entire_binding()),
                            buffer_entry(7, acceleration.material_buffer.as_entire_binding()),
                            buffer_entry(8, acceleration.light_buffer.as_entire_binding()),
                            buffer_entry(9, acceleration.bvh_buffer.as_entire_binding()),
                        ],
                    }),
            };

        if use_wavefront_min {
            let wavefront_workgroups = u32::try_from(pixel_count)
                .map_err(|_| BackendError::Readback("pixel count exceeds u32".to_string()))?
                .div_ceil(64);
            return self.render_wavefront(
                scene,
                request,
                pixel_bounds,
                pixel_count,
                output_size,
                wavefront_workgroups,
                &output_buffer,
                &readback_buffer,
                &control_readback_buffer,
                &wavefront_buffer,
                &bind_group,
            );
        }

        let mut encoder =
            self.device_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 WebGPU dispatch"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pbrt-r4 WebGPU render pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(self.pipeline.stage(ShaderStageId::LegacyRender));
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
        self.device_context.queue.submit(Some(encoder.finish()));

        let mapped = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        mapped.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result.map_err(|error| error.to_string()));
        });
        self.device_context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| BackendError::Readback(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| BackendError::Readback(error.to_string()))?
            .map_err(BackendError::Readback)?;
        let bytes = mapped
            .get_mapped_range()
            .map_err(|error| BackendError::Readback(error.to_string()))?;
        let pixels = bytemuck::try_cast_slice::<u8, [f32; 4]>(&bytes)
            .map_err(|error| BackendError::Readback(error.to_string()))?;
        let rgb = pixels
            .iter()
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect::<Vec<_>>();
        drop(bytes);
        readback_buffer.unmap();
        GpuRenderOutput::new(pixel_bounds, rgb.into_boxed_slice(), request)
            .map_err(|error| BackendError::Readback(format!("invalid render output: {error:?}")))
    }

    #[allow(clippy::too_many_arguments)]
    fn render_wavefront(
        &self,
        scene: &ExecutableScene,
        request: GpuRenderRequest,
        pixel_bounds: super::super::ir::GpuBounds2i,
        pixel_count: usize,
        output_size: wgpu::BufferAddress,
        wavefront_workgroups: u32,
        output_buffer: &wgpu::Buffer,
        readback_buffer: &wgpu::Buffer,
        control_readback_buffer: &wgpu::Buffer,
        wavefront_buffer: &wgpu::Buffer,
        bind_group: &wgpu::BindGroup,
    ) -> Result<GpuRenderOutput, BackendError> {
        let mut sample_offset = 0u32;
        let mut samples_since_readback = 0u32;
        let mut last_readback = Instant::now();
        let mut latest_rgb = None;

        while sample_offset < request.sample_count {
            let mut encoder = self.device_context.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 WebGPU wavefront dispatch"),
                },
            );
            if sample_offset == 0 {
                encoder.clear_buffer(output_buffer, 0, None);
                encoder.clear_buffer(wavefront_buffer, 0, None);
            }
            let mut batch_count = 0u32;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pbrt-r4 WebGPU wavefront pass"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, bind_group, &[]);
                while sample_offset < request.sample_count {
                    pass.set_pipeline(self.pipeline.stage(ShaderStageId::PrepareCameraRays));
                    pass.dispatch_workgroups(1, 1, 1);
                    pass.set_pipeline(self.pipeline.stage(ShaderStageId::GenerateCameraRays));
                    pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                    for _ in 0..=scene.scene().render.integrator.max_depth {
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::IntersectClosest));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::ClassifyIntersection));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(
                            self.pipeline
                                .stage(ShaderStageId::EvaluateSurfaceInteraction),
                        );
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::EvaluateMaterial));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::RegisterBxdf));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::CountBxdf));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::PartitionBxdf));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::SampleDirectLighting));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::GenerateIndirectRays));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::HandleEscapedRays));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(
                            self.pipeline
                                .stage(ShaderStageId::HandleEmissiveIntersection),
                        );
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::IntersectShadow));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::FinishBounce));
                        pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::PrepareNextBounce));
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    pass.set_pipeline(self.pipeline.stage(ShaderStageId::UpdateFilm));
                    pass.dispatch_workgroups(wavefront_workgroups, 1, 1);
                    if sample_offset + 1 < request.sample_count {
                        pass.set_pipeline(self.pipeline.stage(ShaderStageId::AdvanceSample));
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    sample_offset += 1;
                    batch_count += 1;
                    samples_since_readback += 1;
                    if sample_offset == request.sample_count
                        || samples_since_readback >= WAVEFRONT_READBACK_SPP_INTERVAL
                        || last_readback.elapsed() >= WAVEFRONT_READBACK_TIME_INTERVAL
                    {
                        break;
                    }
                }
            }
            encoder.copy_buffer_to_buffer(output_buffer, 0, readback_buffer, 0, output_size);
            encoder.copy_buffer_to_buffer(
                wavefront_buffer,
                0,
                control_readback_buffer,
                0,
                u64::from(super::wavefront::WAVEFRONT_CONTROL_SIZE),
            );
            self.device_context.queue.submit(Some(encoder.finish()));
            latest_rgb =
                Some(self.readback_film(readback_buffer, control_readback_buffer, pixel_count)?);
            samples_since_readback = 0;
            last_readback = Instant::now();
            debug_assert!(batch_count > 0);
        }

        let rgb = latest_rgb.ok_or_else(|| {
            BackendError::Readback("wavefront produced no film readback".to_string())
        })?;
        GpuRenderOutput::new(pixel_bounds, rgb.into_boxed_slice(), request)
            .map_err(|error| BackendError::Readback(format!("invalid render output: {error:?}")))
    }

    fn readback_film(
        &self,
        readback_buffer: &wgpu::Buffer,
        control_readback_buffer: &wgpu::Buffer,
        pixel_count: usize,
    ) -> Result<Vec<[f32; 3]>, BackendError> {
        let mapped = readback_buffer.slice(..);
        let control_mapped = control_readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(2);
        let output_sender = sender.clone();
        mapped.map_async(wgpu::MapMode::Read, move |result| {
            let _ = output_sender.send((0u8, result.map_err(|error| error.to_string())));
        });
        control_mapped.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send((1u8, result.map_err(|error| error.to_string())));
        });
        self.device_context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| BackendError::Readback(error.to_string()))?;
        for _ in 0..2 {
            let (_, result) = receiver
                .recv()
                .map_err(|error| BackendError::Readback(error.to_string()))?;
            result.map_err(BackendError::Readback)?;
        }
        let control_bytes = control_mapped
            .get_mapped_range()
            .map_err(|error| BackendError::Readback(error.to_string()))?;
        let control = bytemuck::try_cast_slice::<u8, u32>(&control_bytes)
            .map_err(|error| BackendError::Readback(error.to_string()))?;
        let overflow = control.get(3).copied().unwrap_or_default() != 0;
        drop(control_bytes);
        control_readback_buffer.unmap();
        if overflow {
            readback_buffer.unmap();
            return Err(BackendError::Readback(
                "WebGPU wavefront queue overflow".to_string(),
            ));
        }
        let bytes = mapped
            .get_mapped_range()
            .map_err(|error| BackendError::Readback(error.to_string()))?;
        let pixels = bytemuck::try_cast_slice::<u8, [f32; 4]>(&bytes)
            .map_err(|error| BackendError::Readback(error.to_string()))?;
        let rgb = pixels
            .iter()
            .take(pixel_count)
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect();
        drop(bytes);
        readback_buffer.unmap();
        Ok(rgb)
    }

    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.device_context.adapter.get_info()
    }

    pub fn acceleration_mode(&self) -> AccelerationMode {
        self.device_context.acceleration_mode
    }

    pub fn max_texture_dimension_2d(&self) -> u32 {
        self.device_context.max_texture_dimension_2d
    }
}

fn buffer_entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn create_pipeline(context: &DeviceContext, mode: AccelerationMode) -> Pipeline {
    let shader_set = build_shader_set(mode).expect("built-in WebGPU shader recipe is valid");
    let bind_group_layout = match mode {
        AccelerationMode::HardwareRayQuery => create_hardware_bind_group_layout(&context.device),
        AccelerationMode::SoftwareBvh => create_software_bind_group_layout(&context.device),
    };
    let shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_set.label),
            source: wgpu::ShaderSource::Wgsl(shader_set.source.into()),
        });
    let pipeline_layout = context
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbrt-r4 WebGPU pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
    let stages = shader_set
        .stages
        .iter()
        .map(|stage| {
            let pipeline =
                context
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(stage.entry_point),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: Some(stage.entry_point),
                        compilation_options: Default::default(),
                        cache: None,
                    });
            (stage.id, pipeline)
        })
        .collect();
    Pipeline {
        stages,
        bind_group_layout,
    }
}

fn create_hardware_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries = common_bind_group_entries();
    entries.insert(
        2,
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::AccelerationStructure {
                vertex_return: false,
            },
            count: None,
        },
    );
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 10,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pbrt-r4 WebGPU hardware bind group layout"),
        entries: &entries,
    })
}

fn create_software_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries = common_bind_group_entries();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pbrt-r4 WebGPU software BVH bind group layout"),
        entries: &entries,
    })
}

fn common_bind_group_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let storage_read_only = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        storage_read_only(3),
        storage_read_only(4),
        storage_read_only(5),
        storage_read_only(6),
        storage_read_only(7),
        storage_read_only(8),
    ];
    // Binding 9 is the texture table for hardware mode and the texture table
    // plus BVH data for software mode.
    entries.push(storage_read_only(9));
    entries
}

fn camera_uniform_bytes(
    scene: GpuSceneView<'_>,
    request: GpuRenderRequest,
    bvh_primitive_offset: u32,
    bvh_node_offset: u32,
) -> Vec<u8> {
    let camera_transform = match scene
        .transforms
        .get(scene.render.camera.render_from_camera.0 as usize)
    {
        Some(super::super::ir::GpuTransform::Static(transform)) => transform.render_from_object,
        _ => GpuMatrix4x4::identity(),
    };
    let pixel_bounds = scene.render.film.pixel_bounds;
    let width = (pixel_bounds.max[0] - pixel_bounds.min[0]) as f32;
    let height = (pixel_bounds.max[1] - pixel_bounds.min[1]) as f32;
    let mut values = Vec::with_capacity(13 * 16);
    for matrix in [scene.render.camera.camera_from_raster, camera_transform] {
        super::geometry::append_wgsl_matrix(&mut values, matrix.0);
    }
    values.extend(
        [
            pixel_bounds.min[0] as f32,
            pixel_bounds.min[1] as f32,
            width,
            height,
        ]
        .into_iter()
        .flat_map(f32::to_ne_bytes),
    );
    values.extend(
        [
            bvh_primitive_offset,
            bvh_node_offset,
            scene.render.integrator.max_depth,
            0,
        ]
        .into_iter()
        .flat_map(u32::to_ne_bytes),
    );
    let seed = scene.render.sampler.seed;
    values.extend(
        [
            seed as u32,
            (seed >> 32) as u32,
            request.sample_start as u32,
            request.sample_count,
        ]
        .into_iter()
        .flat_map(u32::to_ne_bytes),
    );
    values.extend(
        [
            scene.render.filter.radius.0[0],
            scene.render.filter.radius.0[1],
            0.0,
            0.0,
        ]
        .into_iter()
        .flat_map(f32::to_ne_bytes),
    );
    values.extend(
        [
            scene.render.camera.lens_radius,
            scene.render.camera.focal_distance,
            scene.render.camera.shutter_open,
            scene.render.camera.shutter_close,
        ]
        .into_iter()
        .flat_map(f32::to_ne_bytes),
    );
    values
}
