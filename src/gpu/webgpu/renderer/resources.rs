use super::super::super::ir::{GpuBounds2i, GpuSceneView};
use super::super::arena::ArenaLayout;
use super::super::error::BackendError;
use super::{Pipeline, SceneResources};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

pub struct RenderDimensions {
    pub pixel_bounds: GpuBounds2i,
    pub pixel_count: usize,
    pub output_size: wgpu::BufferAddress,
    pub output_buffer_size: wgpu::BufferAddress,
    pub arena_layout: ArenaLayout,
    pub workgroups: u32,
}

impl RenderDimensions {
    pub fn from_scene(scene: GpuSceneView<'_>) -> Result<Self, BackendError> {
        let pixel_bounds = scene.render.film.pixel_bounds;
        let width = pixel_bounds.max[0]
            .checked_sub(pixel_bounds.min[0])
            .ok_or_else(invalid_pixel_bounds)?;
        let height = pixel_bounds.max[1]
            .checked_sub(pixel_bounds.min[1])
            .ok_or_else(invalid_pixel_bounds)?;
        if width == 0 || height == 0 {
            return Err(invalid_pixel_bounds());
        }
        let pixel_count = usize::try_from(width)
            .ok()
            .and_then(|width| usize::try_from(height).ok()?.checked_mul(width))
            .ok_or_else(|| BackendError::Readback("pixel count overflow".to_string()))?;
        let output_size = wgpu::BufferAddress::try_from(
            pixel_count
                .checked_mul(std::mem::size_of::<[f32; 4]>())
                .ok_or_else(|| BackendError::Readback("output size overflow".to_string()))?,
        )
        .map_err(|_| BackendError::Readback("output size overflow".to_string()))?;
        let output_buffer_size = output_size;
        let pixel_count_u32 = u32::try_from(pixel_count)
            .map_err(|_| BackendError::Readback("pixel count exceeds u32".to_string()))?;
        let arena_layout = ArenaLayout::for_pixel_count(pixel_count_u32)
            .ok_or_else(|| BackendError::Readback("wavefront arena size overflow".to_string()))?;
        Ok(Self {
            pixel_bounds,
            pixel_count,
            output_size,
            output_buffer_size,
            workgroups: pixel_count_u32.div_ceil(64),
            arena_layout,
        })
    }
}

pub struct RenderBuffers {
    pub output: wgpu::Buffer,
    pub readback: wgpu::Buffer,
    pub arena: wgpu::Buffer,
}

impl RenderBuffers {
    pub fn create(device: &wgpu::Device, dimensions: &RenderDimensions) -> Self {
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbrt-r4 WebGPU film output"),
            size: dimensions.output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbrt-r4 WebGPU film readback"),
            size: dimensions.output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbrt-r4 WebGPU wavefront arena"),
            size: u64::from(dimensions.arena_layout.byte_len),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            output,
            readback,
            arena,
        }
    }
}

pub fn create_uniform_buffer(
    device: &wgpu::Device,
    scene: GpuSceneView<'_>,
    request: super::super::super::ir::GpuRenderRequest,
    bvh_primitive_offset: u32,
    bvh_node_offset: u32,
) -> wgpu::Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some("pbrt-r4 WebGPU camera uniforms"),
        contents: &super::camera_uniform_bytes(
            scene,
            request,
            bvh_primitive_offset,
            bvh_node_offset,
        ),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

pub fn create_bind_group(
    device: &wgpu::Device,
    pipeline: &Pipeline,
    uniform: &wgpu::Buffer,
    buffers: &RenderBuffers,
    resources: &SceneResources,
) -> wgpu::BindGroup {
    let entries = match resources {
        SceneResources::Hardware(acceleration) => vec![
            buffer_entry(0, uniform.as_entire_binding()),
            buffer_entry(1, buffers.output.as_entire_binding()),
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::AccelerationStructure(&acceleration.tlas),
            },
            buffer_entry(3, acceleration.vertex_buffer.as_entire_binding()),
            buffer_entry(4, acceleration.index_buffer.as_entire_binding()),
            buffer_entry(5, acceleration.primitive_buffer.as_entire_binding()),
            buffer_entry(6, acceleration.transform_buffer.as_entire_binding()),
            buffer_entry(7, acceleration.material_buffer.as_entire_binding()),
            buffer_entry(8, acceleration.light_buffer.as_entire_binding()),
            buffer_entry(9, acceleration.texture_buffer.as_entire_binding()),
            buffer_entry(10, buffers.arena.as_entire_binding()),
        ],
    };
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(match resources {
            SceneResources::Hardware(_) => "pbrt-r4 WebGPU hardware bind group",
        }),
        layout: &pipeline.bind_group_layout,
        entries: &entries,
    })
}

fn buffer_entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn invalid_pixel_bounds() -> BackendError {
    BackendError::Readback("pixel bounds must contain at least one pixel".to_string())
}
