use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::abi::{QueueState, RayWorkItem, SurfaceWorkItem};
use crate::util::error::PbrtError;

pub struct QueueBuffer {
    pub items: wgpu::Buffer,
    pub state: wgpu::Buffer,
    pub capacity: u32,
}

impl QueueBuffer {
    pub fn new<T: Pod>(
        device: &wgpu::Device,
        label: &str,
        capacity: u32,
    ) -> Result<Self, PbrtError> {
        if capacity == 0 {
            return Err(PbrtError::error(&format!(
                "WebGPU queue \"{label}\" capacity must be positive."
            )));
        }
        let item_size = u64::try_from(std::mem::size_of::<T>())
            .map_err(|_| PbrtError::error("WebGPU queue item size does not fit in u64."))?;
        let size = item_size.checked_mul(u64::from(capacity)).ok_or_else(|| {
            PbrtError::error(&format!("WebGPU queue \"{label}\" size overflowed."))
        })?;
        if size == 0 {
            return Err(PbrtError::error(&format!(
                "WebGPU queue \"{label}\" item size must be positive."
            )));
        }
        let state = QueueState {
            count: 0,
            capacity,
            overflow: 0,
            padding: 0,
        };
        Ok(Self {
            items: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("pbrt-r4 {label} items")),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            state: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("pbrt-r4 {label} state")),
                contents: bytemuck::bytes_of(&state),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
            capacity,
        })
    }

    pub fn reset(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.state, 0, Some(4));
        encoder.clear_buffer(&self.state, 8, Some(4));
    }
}

pub struct Queues {
    pub rays: wgpu::Buffer,
    pub surfaces: wgpu::Buffer,
    pub camera_rays: QueueBuffer,
    pub current_rays: QueueBuffer,
    pub next_rays: QueueBuffer,
    pub shadow_rays: QueueBuffer,
    pub escaped_rays: QueueBuffer,
    pub hit_area_lights: QueueBuffer,
    pub material_evals: QueueBuffer,
}

impl Queues {
    pub fn new(device: &wgpu::Device, pixel_count: u64) -> Result<Self, PbrtError> {
        let ray_size = pixel_count
            .checked_mul(2)
            .and_then(|size| size.checked_mul(std::mem::size_of::<RayWorkItem>() as u64))
            .ok_or_else(|| PbrtError::error("WebGPU ray queue size overflowed."))?;
        let surface_size = pixel_count
            .checked_mul(std::mem::size_of::<SurfaceWorkItem>() as u64)
            .ok_or_else(|| PbrtError::error("WebGPU surface queue size overflowed."))?;
        let capacity = u32::try_from(pixel_count)
            .map_err(|_| PbrtError::error("WebGPU queue capacity does not fit in u32."))?;
        Ok(Self {
            rays: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 primary ray queue"),
                size: ray_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            surfaces: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 surface work buffer"),
                size: surface_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            camera_rays: QueueBuffer::new::<RayWorkItem>(device, "camera ray queue", capacity)?,
            current_rays: QueueBuffer::new::<RayWorkItem>(device, "current ray queue", capacity)?,
            next_rays: QueueBuffer::new::<RayWorkItem>(device, "next ray queue", capacity)?,
            shadow_rays: QueueBuffer::new::<SurfaceWorkItem>(device, "shadow ray queue", capacity)?,
            escaped_rays: QueueBuffer::new::<RayWorkItem>(device, "escaped ray queue", capacity)?,
            hit_area_lights: QueueBuffer::new::<SurfaceWorkItem>(
                device,
                "hit area light queue",
                capacity,
            )?,
            material_evals: QueueBuffer::new::<SurfaceWorkItem>(
                device,
                "material evaluation queue",
                capacity,
            )?,
        })
    }
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct _AbiMarker {
    _value: [u32; 4],
}
