use super::super::compiler::GpuCompiledScene;
use super::super::ir::{GpuMatrix4x4, GpuRenderOutput, GpuRenderRequest, GpuSceneView};
use super::device::{AccelerationMode, DeviceContext, PrepareOptions};
use super::error::BackendError;
use super::geometry::{HardwareAcceleration, ScenePlan};
use super::shader::{build_wavefront_shader_set, ShaderSet, ShaderStageId};
use crate::base::film::Film;

mod resources;
mod wavefront;

use resources::{RenderBuffers, RenderDimensions};

enum SceneResources {
    Hardware(HardwareAcceleration),
}

pub struct ExecutableScene {
    scene: GpuCompiledScene,
    resources: SceneResources,
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
        if options.acceleration_mode != AccelerationMode::HardwareRayQuery {
            return Err(BackendError::UnsupportedAccelerationMode(
                options.acceleration_mode,
            ));
        }
        let device_context = DeviceContext::create(options)?;
        let pipeline = create_pipeline(&device_context);
        Ok(Self {
            device_context,
            pipeline,
        })
    }

    pub fn prepare(&mut self, scene: &GpuCompiledScene) -> Result<ExecutableScene, BackendError> {
        let plan = ScenePlan::from_scene(scene.view())?;
        let resources = match self.device_context.acceleration_mode {
            AccelerationMode::HardwareRayQuery => {
                SceneResources::Hardware(HardwareAcceleration::create(&self.device_context, &plan)?)
            }
            AccelerationMode::SoftwareBvh => {
                unreachable!("software mode is rejected in Renderer::new")
            }
        };
        Ok(ExecutableScene {
            scene: scene.clone(),
            resources,
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
        let dimensions = RenderDimensions::from_scene(scene_view)?;

        let (bvh_primitive_offset, bvh_node_offset) = (0, 0);
        let uniform_buffer = resources::create_uniform_buffer(
            &self.device_context.device,
            scene_view,
            request,
            bvh_primitive_offset,
            bvh_node_offset,
        );
        let buffers = RenderBuffers::create(&self.device_context.device, &dimensions);
        let bind_group = resources::create_bind_group(
            &self.device_context.device,
            &self.pipeline,
            &uniform_buffer,
            &buffers,
            &scene.resources,
        );

        wavefront::render(self, request, &dimensions, &buffers, &bind_group)
    }

    pub fn render_to_film(
        &mut self,
        scene: &ExecutableScene,
        request: &GpuRenderRequest,
        film: &mut Film,
    ) -> Result<(), BackendError> {
        let output = self.render(scene, request)?;
        output.write_to_film(film);
        film.write_image();
        Ok(())
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

fn create_pipeline(context: &DeviceContext) -> Pipeline {
    let shader_set =
        build_wavefront_shader_set().expect("built-in WebGPU wavefront recipe is valid");
    create_pipeline_from_shader_set(context, AccelerationMode::HardwareRayQuery, shader_set)
}

fn create_pipeline_from_shader_set(
    context: &DeviceContext,
    mode: AccelerationMode,
    shader_set: ShaderSet,
) -> Pipeline {
    assert_eq!(mode, AccelerationMode::HardwareRayQuery);
    let bind_group_layout = create_hardware_bind_group_layout(&context.device);
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
