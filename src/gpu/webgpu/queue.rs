use bytemuck::{Pod, Zeroable};

use super::abi::SurfaceWorkItem;
use crate::util::error::PbrtError;

pub struct Queues {
    pub surfaces: wgpu::Buffer,
    pub wavefront: wgpu::Buffer,
}

impl Queues {
    pub fn new(device: &wgpu::Device, pixel_count: u64) -> Result<Self, PbrtError> {
        let surface_size = pixel_count
            .checked_mul(std::mem::size_of::<SurfaceWorkItem>() as u64)
            .ok_or_else(|| PbrtError::error("WebGPU surface queue size overflowed."))?;
        // The packed queue contains two RayWorkItem arrays preceded by eight
        // u32 words for the current/next counters and overflow flags.
        let wavefront_words = 16u64
            .checked_add(pixel_count.checked_mul(32).ok_or_else(|| {
                PbrtError::error("WebGPU packed wavefront queue size overflowed.")
            })?)
            .ok_or_else(|| PbrtError::error("WebGPU packed wavefront queue size overflowed."))?;
        let wavefront_size = wavefront_words
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or_else(|| PbrtError::error("WebGPU packed wavefront queue size overflowed."))?;
        let _capacity = u32::try_from(pixel_count)
            .map_err(|_| PbrtError::error("WebGPU queue capacity does not fit in u32."))?;
        Ok(Self {
            surfaces: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 surface work buffer"),
                size: surface_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            wavefront: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 packed wavefront ray queue"),
                size: wavefront_size,
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
