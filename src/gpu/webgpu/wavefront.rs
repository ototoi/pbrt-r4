//! Host-side layout for the initial fixed-slot GPU wavefront arena.

use std::mem::{align_of, size_of};

// Ten ray/path vec4 records plus nine surface/material/path-history records.
pub const WAVEFRONT_SLOT_STRIDE: u32 = 304;
pub const WAVEFRONT_CONTROL_SIZE: u32 = 16;
pub const WAVEFRONT_QUEUE_HEADER_COUNT: u32 = 6;
pub const WAVEFRONT_QUEUE_HEADER_SIZE: u32 = QUEUE_HEADER_STRIDE;
pub const WAVEFRONT_ARENA_HEADER_SIZE: u32 =
    WAVEFRONT_CONTROL_SIZE + WAVEFRONT_QUEUE_HEADER_COUNT * WAVEFRONT_QUEUE_HEADER_SIZE;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WavefrontControl {
    pub sample_index: u32,
    pub active_count: u32,
    pub next_count: u32,
    pub overflow: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueueHeader {
    pub count: u32,
    pub capacity: u32,
    pub offset: u32,
    pub overflow: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IntersectionRecord {
    pub state: u32,
    pub primitive_id: u32,
    pub triangle_id: u32,
    pub _padding: u32,
    pub barycentrics: [f32; 4],
    pub t: f32,
    pub _padding2: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfaceInteractionRecord {
    pub position: [f32; 4],
    pub position_error: [f32; 4],
    pub geometric_normal: [f32; 4],
    pub shading_normal: [f32; 4],
    pub uv: [f32; 4],
    pub dpdu: [f32; 4],
    pub dpdv: [f32; 4],
    pub wo: [f32; 4],
    pub primitive_id: u32,
    pub material_id: u32,
    pub area_light_source: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BxDFWorkItem {
    pub surface_index: u32,
    pub bxdf_offset: u32,
    pub bxdf_count: u32,
    pub path_state: u32,
    pub throughput: [f32; 4],
    pub previous_bsdf_pdf: f32,
    pub previous_light_pdf: f32,
    pub depth: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DirectLightingContribution {
    pub radiance: [f32; 4],
    pub pixel_index: u32,
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShadowWorkItem {
    pub origin: [f32; 4],
    pub direction: [f32; 4],
    pub max_t: f32,
    pub pixel_index: u32,
    pub source_primitive: u32,
    pub source_triangle: u32,
    pub contribution: [f32; 4],
}

pub const QUEUE_HEADER_STRIDE: u32 = size_of::<QueueHeader>() as u32;
pub const INTERSECTION_RECORD_STRIDE: u32 = size_of::<IntersectionRecord>() as u32;
pub const SURFACE_INTERACTION_RECORD_STRIDE: u32 = size_of::<SurfaceInteractionRecord>() as u32;
pub const BXDF_WORK_ITEM_STRIDE: u32 = size_of::<BxDFWorkItem>() as u32;
pub const DIRECT_LIGHTING_CONTRIBUTION_STRIDE: u32 = size_of::<DirectLightingContribution>() as u32;
pub const SHADOW_WORK_ITEM_STRIDE: u32 = size_of::<ShadowWorkItem>() as u32;

pub fn validate_wavefront_abi() -> bool {
    align_of::<WavefrontControl>() == 4
        && size_of::<WavefrontControl>() == WAVEFRONT_CONTROL_SIZE as usize
        && align_of::<QueueHeader>() == 4
        && QUEUE_HEADER_STRIDE == 16
        && align_of::<IntersectionRecord>() == 4
        && INTERSECTION_RECORD_STRIDE == 48
        && align_of::<SurfaceInteractionRecord>() == 4
        && SURFACE_INTERACTION_RECORD_STRIDE == 144
        && align_of::<BxDFWorkItem>() == 4
        && BXDF_WORK_ITEM_STRIDE == 48
        && align_of::<DirectLightingContribution>() == 4
        && DIRECT_LIGHTING_CONTRIBUTION_STRIDE == 32
        && align_of::<ShadowWorkItem>() == 4
        && SHADOW_WORK_ITEM_STRIDE == 64
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WavefrontLayout {
    pub capacity: u32,
    pub byte_len: u32,
}

impl WavefrontLayout {
    pub fn for_pixel_count(pixel_count: u32) -> Option<Self> {
        let byte_len = WAVEFRONT_SLOT_STRIDE
            .checked_mul(pixel_count)?
            .checked_add(WAVEFRONT_ARENA_HEADER_SIZE)?;
        Some(Self {
            capacity: pixel_count,
            byte_len,
        })
    }

    pub fn slot_offset(self, index: u32) -> Option<u32> {
        (index < self.capacity).then(|| WAVEFRONT_ARENA_HEADER_SIZE + index * WAVEFRONT_SLOT_STRIDE)
    }
}
