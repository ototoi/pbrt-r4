use bytemuck::{bytes_of, Zeroable};
use wgpu::util::DeviceExt;

use super::abi::{
    AttributesEvalWorkItem, DirectLightSample, DispatchIndirectArgs, PixelSampleState,
    QueueCounters, QueueState, RayWorkItem, RenderError, ShadowRayWorkItem, SurfaceWorkItem,
    TextureEvalResult, QUEUE_DISPATCH_SLOT_COUNT,
};
use crate::util::error::PbrtError;

const QUEUE_COUNT: u64 = 14;
const QUEUE_COUNTER_BYTES: u64 = QUEUE_COUNT * std::mem::size_of::<QueueState>() as u64;
const RENDER_ERROR_BYTES: u64 = std::mem::size_of::<RenderError>() as u64;
const STATE_READBACK_BYTES: u64 = QUEUE_COUNTER_BYTES + RENDER_ERROR_BYTES;
const QUEUE_DISPATCH_ARGS_BYTES: u64 =
    QUEUE_DISPATCH_SLOT_COUNT * std::mem::size_of::<DispatchIndirectArgs>() as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypedQueueSizes {
    pub surfaces: u64,
    pub pixel_sample_states: u64,
    pub current_rays: u64,
    pub next_rays: u64,
    pub shadow_rays: u64,
    pub material_ray_indices: u64,
    pub attributes_eval_work_items: u64,
    pub texture_eval_results: u64,
    pub hit_area_ray_indices: u64,
    pub escaped_ray_indices: u64,
    pub direct_light_samples: u64,
    pub direct_eval_ray_indices: u64,
    pub scatter_diffuse_ray_indices: u64,
    pub scatter_diffuse_transmission_ray_indices: u64,
    pub scatter_conductor_ray_indices: u64,
    pub scatter_dielectric_ray_indices: u64,
    pub scatter_thin_dielectric_ray_indices: u64,
    pub scatter_measured_ray_indices: u64,
    pub scatter_coated_ray_indices: u64,
}

impl TypedQueueSizes {
    pub fn new(
        pixel_count: u64,
        attributes_eval_stride: u64,
        texture_eval_stride: u64,
    ) -> Result<Self, PbrtError> {
        u32::try_from(pixel_count)
            .map_err(|_| PbrtError::error("WebGPU pixel count does not fit queue indices."))?;
        if attributes_eval_stride == 0 {
            return Err(PbrtError::error(
                "WebGPU attributes eval stride must be greater than zero.",
            ));
        }
        if texture_eval_stride == 0 {
            return Err(PbrtError::error(
                "WebGPU texture eval stride must be greater than zero.",
            ));
        }
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
            attributes_eval_work_items: bytes(
                std::mem::size_of::<AttributesEvalWorkItem>(),
                "evaluated attributes",
            )?
            .checked_mul(attributes_eval_stride)
            .ok_or_else(|| PbrtError::error("WebGPU evaluated attributes size overflowed."))?,
            texture_eval_results: bytes(
                std::mem::size_of::<TextureEvalResult>(),
                "texture evaluation results",
            )?
            .checked_mul(texture_eval_stride)
            .ok_or_else(|| PbrtError::error("WebGPU texture results size overflowed."))?,
            hit_area_ray_indices: bytes(std::mem::size_of::<u32>(), "hit-area queue")?,
            escaped_ray_indices: bytes(std::mem::size_of::<u32>(), "escaped queue")?,
            direct_light_samples: bytes(
                std::mem::size_of::<DirectLightSample>(),
                "direct light samples",
            )?,
            direct_eval_ray_indices: bytes(std::mem::size_of::<u32>(), "direct-eval queue")?,
            scatter_diffuse_ray_indices: bytes(
                std::mem::size_of::<u32>(),
                "scatter diffuse queue",
            )?,
            scatter_diffuse_transmission_ray_indices: bytes(
                std::mem::size_of::<u32>(),
                "scatter diffuse transmission queue",
            )?,
            scatter_conductor_ray_indices: bytes(
                std::mem::size_of::<u32>(),
                "scatter conductor queue",
            )?,
            scatter_dielectric_ray_indices: bytes(
                std::mem::size_of::<u32>(),
                "scatter dielectric queue",
            )?,
            scatter_thin_dielectric_ray_indices: bytes(
                std::mem::size_of::<u32>(),
                "scatter thin dielectric queue",
            )?,
            scatter_measured_ray_indices: bytes(
                std::mem::size_of::<u32>(),
                "scatter measured queue",
            )?,
            scatter_coated_ray_indices: bytes(std::mem::size_of::<u32>(), "scatter coated queue")?,
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
    pub attributes_eval_work_items: wgpu::Buffer,
    pub texture_eval_results: wgpu::Buffer,
    pub hit_area_ray_indices: wgpu::Buffer,
    pub escaped_ray_indices: wgpu::Buffer,
    pub direct_light_samples: wgpu::Buffer,
    pub direct_eval_ray_indices: wgpu::Buffer,
    pub scatter_diffuse_ray_indices: wgpu::Buffer,
    pub scatter_diffuse_transmission_ray_indices: wgpu::Buffer,
    pub scatter_conductor_ray_indices: wgpu::Buffer,
    pub scatter_dielectric_ray_indices: wgpu::Buffer,
    pub scatter_thin_dielectric_ray_indices: wgpu::Buffer,
    pub scatter_measured_ray_indices: wgpu::Buffer,
    pub scatter_coated_ray_indices: wgpu::Buffer,
    pub queue_dispatch_args: wgpu::Buffer,
    state_readback: wgpu::Buffer,
}

impl Queues {
    pub fn new(
        device: &wgpu::Device,
        pixel_count: u64,
        attributes_eval_stride: u64,
        texture_eval_stride: u64,
    ) -> Result<Self, PbrtError> {
        let sizes = TypedQueueSizes::new(pixel_count, attributes_eval_stride, texture_eval_stride)?;
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
            direct: state,
            scatter_diffuse: state,
            scatter_diffuse_transmission: state,
            scatter_conductor: state,
            scatter_dielectric: state,
            scatter_thin_dielectric: state,
            scatter_measured: state,
            scatter_coated: state,
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
            attributes_eval_work_items: storage(
                "pbrt-r4 evaluated material attributes",
                sizes.attributes_eval_work_items,
            ),
            texture_eval_results: storage(
                "pbrt-r4 texture evaluation results",
                sizes.texture_eval_results,
            ),
            hit_area_ray_indices: storage(
                "pbrt-r4 hit-area ray index queue",
                sizes.hit_area_ray_indices,
            ),
            escaped_ray_indices: storage(
                "pbrt-r4 escaped ray index queue",
                sizes.escaped_ray_indices,
            ),
            direct_light_samples: storage(
                "pbrt-r4 direct light samples",
                sizes.direct_light_samples,
            ),
            direct_eval_ray_indices: storage(
                "pbrt-r4 direct-eval ray index queue",
                sizes.direct_eval_ray_indices,
            ),
            scatter_diffuse_ray_indices: storage(
                "pbrt-r4 scatter diffuse ray index queue",
                sizes.scatter_diffuse_ray_indices,
            ),
            scatter_diffuse_transmission_ray_indices: storage(
                "pbrt-r4 scatter diffuse transmission ray index queue",
                sizes.scatter_diffuse_transmission_ray_indices,
            ),
            scatter_conductor_ray_indices: storage(
                "pbrt-r4 scatter conductor ray index queue",
                sizes.scatter_conductor_ray_indices,
            ),
            scatter_dielectric_ray_indices: storage(
                "pbrt-r4 scatter dielectric ray index queue",
                sizes.scatter_dielectric_ray_indices,
            ),
            scatter_thin_dielectric_ray_indices: storage(
                "pbrt-r4 scatter thin dielectric ray index queue",
                sizes.scatter_thin_dielectric_ray_indices,
            ),
            scatter_measured_ray_indices: storage(
                "pbrt-r4 scatter measured ray index queue",
                sizes.scatter_measured_ray_indices,
            ),
            scatter_coated_ray_indices: storage(
                "pbrt-r4 scatter coated ray index queue",
                sizes.scatter_coated_ray_indices,
            ),
            queue_dispatch_args: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbrt-r4 queue dispatch args"),
                size: QUEUE_DISPATCH_ARGS_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
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
        // Each QueueState is 4 u32 words (count, capacity, overflow, padding);
        // the overflow flag is the third word of every queue in QueueCounters.
        let overflowed = (0..QUEUE_COUNT)
            .map(|index| index as usize * 4 + 2)
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
