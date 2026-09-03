use bytemuck::{Pod, Zeroable};

use super::abi::{HitRecord, RayWorkItem};

pub struct Queues {
    pub rays: wgpu::Buffer,
    pub hits: wgpu::Buffer,
}

impl Queues {
    pub fn new(device: &wgpu::Device, pixel_count: u64) -> Self {
        let ray_size = pixel_count * std::mem::size_of::<RayWorkItem>() as u64;
        let hit_size = pixel_count * std::mem::size_of::<HitRecord>() as u64;
        Self {
            rays: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 primary ray queue"),
                size: ray_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            hits: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 primary hit buffer"),
                size: hit_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        }
    }
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct _AbiMarker {
    _value: [u32; 4],
}
