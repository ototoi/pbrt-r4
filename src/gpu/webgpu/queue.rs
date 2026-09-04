use bytemuck::{Pod, Zeroable};

use super::abi::SurfaceWorkItem;
use crate::util::error::PbrtError;

pub struct Queues {
    pub surfaces: wgpu::Buffer,
    pub wavefront: wgpu::Buffer,
    state_readback: wgpu::Buffer,
}

impl Queues {
    pub fn new(device: &wgpu::Device, pixel_count: u64) -> Result<Self, PbrtError> {
        let surface_size = pixel_count
            .checked_mul(std::mem::size_of::<SurfaceWorkItem>() as u64)
            .ok_or_else(|| PbrtError::error("WebGPU surface queue size overflowed."))?;
        // The packed queue contains two RayWorkItem arrays preceded by eight
        // u32 words for the current/next counters and overflow flags.
        let wavefront_words = 16u64
            .checked_add(pixel_count.checked_mul(33).ok_or_else(|| {
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
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            state_readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 wavefront queue state readback"),
                size: 32,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        })
    }

    pub fn copy_state_to_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.wavefront, 0, &self.state_readback, 0, 32);
    }

    pub fn read_overflow(&self, device: &wgpu::Device) -> Result<bool, PbrtError> {
        let slice = self.state_readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| {
                PbrtError::error(&format!("WebGPU queue-state polling failed: {error}"))
            })?;
        receiver
            .recv()
            .map_err(|error| {
                PbrtError::error(&format!("WebGPU queue-state callback failed: {error}"))
            })?
            .map_err(|error| {
                PbrtError::error(&format!("WebGPU queue-state mapping failed: {error}"))
            })?;
        let mapped = slice.get_mapped_range().map_err(|error| {
            PbrtError::error(&format!("WebGPU queue-state map access failed: {error}"))
        })?;
        let words = bytemuck::try_cast_slice::<u8, u32>(&mapped)
            .map_err(|_| PbrtError::error("WebGPU queue-state readback was not u32-aligned."))?;
        let overflowed = words.get(2).copied().unwrap_or(0) != 0
            || words.get(6).copied().unwrap_or(0) != 0
            || words.get(10).copied().unwrap_or(0) != 0;
        drop(mapped);
        self.state_readback.unmap();
        Ok(overflowed)
    }
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct _AbiMarker {
    _value: [u32; 4],
}
