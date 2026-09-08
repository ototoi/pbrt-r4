use bytemuck::{bytes_of, Zeroable};
use wgpu::util::DeviceExt;

use super::abi::{
    EvaluatedAttributesWorkItem, PixelSampleState, QueueCounters, QueueState, RayWorkItem,
    RenderError, ShadowRayWorkItem, SurfaceWorkItem,
};
use crate::util::error::PbrtError;

const MAX_MATERIAL_TREE_DEPTH: u64 = 3;

const QUEUE_COUNT: u64 = 6;
const QUEUE_COUNTER_BYTES: u64 = QUEUE_COUNT * std::mem::size_of::<QueueState>() as u64;
const RENDER_ERROR_BYTES: u64 = std::mem::size_of::<RenderError>() as u64;
const STATE_READBACK_BYTES: u64 = QUEUE_COUNTER_BYTES + RENDER_ERROR_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypedQueueSizes {
    pub surfaces: u64,
    pub pixel_sample_states: u64,
    pub current_rays: u64,
    pub next_rays: u64,
    pub shadow_rays: u64,
    pub material_ray_indices: u64,
    pub evaluated_attributes: u64,
    pub hit_area_ray_indices: u64,
    pub escaped_ray_indices: u64,
}

impl TypedQueueSizes {
    pub fn new(pixel_count: u64) -> Result<Self, PbrtError> {
        u32::try_from(pixel_count)
            .map_err(|_| PbrtError::error("WebGPU pixel count does not fit queue indices."))?;
        let bytes = |element_size: usize, label: &str| {
            pixel_count
                .checked_mul(element_size as u64)
                .ok_or_else(|| PbrtError::error(&format!("WebGPU {label} size overflowed.")))
        };
        Ok(Self {
            surfaces: bytes(std::mem::size_of::<SurfaceWorkItem>(), "surface buffer")?,
            pixel_sample_states: bytes(
                std::mem::size_of::<PixelSampleState>(),
                "pixel sample state buffer",
            )?,
            current_rays: bytes(std::mem::size_of::<RayWorkItem>(), "current ray buffer")?,
            next_rays: bytes(std::mem::size_of::<RayWorkItem>(), "next ray buffer")?,
            shadow_rays: bytes(
                std::mem::size_of::<ShadowRayWorkItem>(),
                "shadow ray buffer",
            )?,
            material_ray_indices: bytes(std::mem::size_of::<u32>(), "material queue")?,
            evaluated_attributes: bytes(
                std::mem::size_of::<EvaluatedAttributesWorkItem>(),
                "evaluated attributes",
            )?
            .checked_mul(MAX_MATERIAL_TREE_DEPTH)
            .ok_or_else(|| PbrtError::error("WebGPU evaluated attributes size overflowed."))?,
            hit_area_ray_indices: bytes(std::mem::size_of::<u32>(), "hit-area queue")?,
            escaped_ray_indices: bytes(std::mem::size_of::<u32>(), "escaped queue")?,
        })
    }
}

pub struct Queues {
    pub surfaces: wgpu::Buffer,
    pub counters: wgpu::Buffer,
    pub render_error: wgpu::Buffer,
    pub pixel_sample_states: wgpu::Buffer,
    pub current_rays: wgpu::Buffer,
    pub next_rays: wgpu::Buffer,
    pub shadow_rays: wgpu::Buffer,
    pub material_ray_indices: wgpu::Buffer,
    pub evaluated_attributes: wgpu::Buffer,
    pub hit_area_ray_indices: wgpu::Buffer,
    pub escaped_ray_indices: wgpu::Buffer,
    state_readback: wgpu::Buffer,
}

impl Queues {
    pub fn new(device: &wgpu::Device, pixel_count: u64) -> Result<Self, PbrtError> {
        let sizes = TypedQueueSizes::new(pixel_count)?;
        let capacity = u32::try_from(pixel_count)
            .map_err(|_| PbrtError::error("WebGPU queue capacity does not fit in u32."))?;
        let state = QueueState {
            count: 0,
            capacity,
            overflow: 0,
            padding: 0,
        };
        let counters = QueueCounters {
            current: state,
            next: state,
            shadow: state,
            material: state,
            hit_area: state,
            escaped: state,
        };
        let storage = |label: &'static str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        Ok(Self {
            surfaces: storage("pbrt-r4 surface work buffer", sizes.surfaces),
            counters: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 wavefront queue counters"),
                contents: bytes_of(&counters),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
            render_error: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 render error"),
                contents: bytes_of(&RenderError::zeroed()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
            pixel_sample_states: storage("pbrt-r4 pixel sample states", sizes.pixel_sample_states),
            current_rays: storage("pbrt-r4 current ray queue", sizes.current_rays),
            next_rays: storage("pbrt-r4 next ray queue", sizes.next_rays),
            shadow_rays: storage("pbrt-r4 shadow ray queue", sizes.shadow_rays),
            material_ray_indices: storage(
                "pbrt-r4 material ray index queue",
                sizes.material_ray_indices,
            ),
            evaluated_attributes: storage(
                "pbrt-r4 evaluated material attributes",
                sizes.evaluated_attributes,
            ),
            hit_area_ray_indices: storage(
                "pbrt-r4 hit-area ray index queue",
                sizes.hit_area_ray_indices,
            ),
            escaped_ray_indices: storage(
                "pbrt-r4 escaped ray index queue",
                sizes.escaped_ray_indices,
            ),
            state_readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 wavefront state readback"),
                size: STATE_READBACK_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        })
    }

    pub fn copy_state_to_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.counters,
            0,
            &self.state_readback,
            0,
            QUEUE_COUNTER_BYTES,
        );
        encoder.copy_buffer_to_buffer(
            &self.render_error,
            0,
            &self.state_readback,
            QUEUE_COUNTER_BYTES,
            RENDER_ERROR_BYTES,
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
        let overflowed = [2usize, 6, 10, 14, 18, 22]
            .into_iter()
            .any(|index| words.get(index).copied().unwrap_or(0) != 0);
        let render_error = words
            .get(QUEUE_COUNTER_BYTES as usize / std::mem::size_of::<u32>())
            .copied()
            .unwrap_or(0)
            != 0;
        if overflowed || render_error {
            log::error!("WebGPU queue counters and render error: {words:?}");
        }
        drop(mapped);
        self.state_readback.unmap();
        Ok(overflowed || render_error)
    }
}
