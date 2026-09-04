use bytemuck::{Pod, Zeroable};

use super::abi::SurfaceWorkItem;
use crate::util::error::PbrtError;

pub const QUEUE_STATE_WORDS: u64 = 24;
const SAMPLE_STATE_WORDS: u64 = 8;
const RAY_WORDS: u64 = 20;
const SHADOW_WORDS: u64 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedWavefrontLayout {
    pub sample_state_offset_words: u64,
    pub ray_data_offset_words: u64,
    pub shadow_data_offset_words: u64,
    pub material_data_offset_words: u64,
    pub hit_area_data_offset_words: u64,
    pub escaped_data_offset_words: u64,
    pub total_words: u64,
}

impl PackedWavefrontLayout {
    pub fn state_readback_size_bytes(&self) -> u64 {
        QUEUE_STATE_WORDS * std::mem::size_of::<u32>() as u64
    }

    pub fn wavefront_size_bytes(&self) -> Result<u64, PbrtError> {
        self.total_words
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or_else(|| PbrtError::error("WebGPU packed wavefront queue size overflowed."))
    }
}

pub fn packed_wavefront_layout(
    pixel_count: u64,
    max_depth: u32,
) -> Result<PackedWavefrontLayout, PbrtError> {
    u32::try_from(pixel_count)
        .map_err(|_| PbrtError::error("WebGPU pixel count does not fit shader u32 offsets."))?;
    let classification_capacity = pixel_count
        .checked_mul(
            u64::from(max_depth)
                .checked_add(1)
                .ok_or_else(|| PbrtError::error("WebGPU classification depth overflowed."))?,
        )
        .ok_or_else(|| PbrtError::error("WebGPU classification queue size overflowed."))?;
    let sample_state_offset_words = QUEUE_STATE_WORDS;
    let ray_data_offset_words = sample_state_offset_words
        .checked_add(
            pixel_count
                .checked_mul(SAMPLE_STATE_WORDS)
                .ok_or_else(|| PbrtError::error("WebGPU pixel sample state size overflowed."))?,
        )
        .ok_or_else(|| PbrtError::error("WebGPU packed wavefront queue size overflowed."))?;
    let shadow_data_offset_words = ray_data_offset_words
        .checked_add(
            pixel_count
                .checked_mul(RAY_WORDS * 2)
                .ok_or_else(|| PbrtError::error("WebGPU ray queue size overflowed."))?,
        )
        .ok_or_else(|| PbrtError::error("WebGPU packed wavefront queue size overflowed."))?;
    let material_data_offset_words = shadow_data_offset_words
        .checked_add(
            pixel_count
                .checked_mul(SHADOW_WORDS)
                .ok_or_else(|| PbrtError::error("WebGPU shadow queue size overflowed."))?,
        )
        .ok_or_else(|| PbrtError::error("WebGPU packed wavefront queue size overflowed."))?;
    let hit_area_data_offset_words = material_data_offset_words
        .checked_add(classification_capacity)
        .ok_or_else(|| PbrtError::error("WebGPU classification queue size overflowed."))?;
    let escaped_data_offset_words = hit_area_data_offset_words
        .checked_add(classification_capacity)
        .ok_or_else(|| PbrtError::error("WebGPU classification queue size overflowed."))?;
    let total_words = escaped_data_offset_words
        .checked_add(classification_capacity)
        .ok_or_else(|| PbrtError::error("WebGPU classification queue size overflowed."))?;
    u32::try_from(total_words).map_err(|_| {
        PbrtError::error("WebGPU packed wavefront queue does not fit shader u32 offsets.")
    })?;
    Ok(PackedWavefrontLayout {
        sample_state_offset_words,
        ray_data_offset_words,
        shadow_data_offset_words,
        material_data_offset_words,
        hit_area_data_offset_words,
        escaped_data_offset_words,
        total_words,
    })
}

pub struct Queues {
    pub surfaces: wgpu::Buffer,
    pub wavefront: wgpu::Buffer,
    state_readback: wgpu::Buffer,
}

impl Queues {
    pub fn new(device: &wgpu::Device, pixel_count: u64, max_depth: u32) -> Result<Self, PbrtError> {
        let surface_size = pixel_count
            .checked_mul(std::mem::size_of::<SurfaceWorkItem>() as u64)
            .ok_or_else(|| PbrtError::error("WebGPU surface queue size overflowed."))?;
        let layout = packed_wavefront_layout(pixel_count, max_depth)?;
        let wavefront_size = layout.wavefront_size_bytes()?;
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
                size: layout.state_readback_size_bytes(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        })
    }

    pub fn copy_state_to_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.wavefront,
            0,
            &self.state_readback,
            0,
            QUEUE_STATE_WORDS * std::mem::size_of::<u32>() as u64,
        );
    }

    pub fn read_error(&self, device: &wgpu::Device) -> Result<bool, PbrtError> {
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
        let errored = words.get(23).copied().unwrap_or(0) != 0
            || words.get(2).copied().unwrap_or(0) != 0
            || words.get(6).copied().unwrap_or(0) != 0
            || words.get(10).copied().unwrap_or(0) != 0
            || words.get(14).copied().unwrap_or(0) != 0
            || words.get(18).copied().unwrap_or(0) != 0
            || words.get(22).copied().unwrap_or(0) != 0;
        drop(mapped);
        self.state_readback.unmap();
        Ok(errored)
    }
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct _AbiMarker {
    _value: [u32; 4],
}
