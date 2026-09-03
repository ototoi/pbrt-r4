use bytemuck::{Pod, Zeroable};

use super::abi::{RayWorkItem, SurfaceWorkItem};
use crate::util::error::PbrtError;

pub struct Queues {
    pub rays: wgpu::Buffer,
    pub surfaces: wgpu::Buffer,
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
        })
    }
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct _AbiMarker {
    _value: [u32; 4],
}
