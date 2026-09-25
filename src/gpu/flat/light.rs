use super::AttributeRef;

pub const INVALID_INDEX: u32 = u32::MAX;

pub const AREA_LIGHT_FLAG_TWO_SIDED: u32 = 1 << 0;
pub const AREA_LIGHT_FLAG_ZERO_ALPHA_SAMPLE_ONLY: u32 = 1 << 1;

pub fn area_light_flags(two_sided: bool, zero_alpha_sample_only: bool) -> u32 {
    (if two_sided {
        AREA_LIGHT_FLAG_TWO_SIDED
    } else {
        0
    }) | (if zero_alpha_sample_only {
        AREA_LIGHT_FLAG_ZERO_ALPHA_SAMPLE_ONLY
    } else {
        0
    })
}

pub fn area_light_is_two_sided(flags: u32) -> bool {
    flags & AREA_LIGHT_FLAG_TWO_SIDED != 0
}

pub fn area_light_is_zero_alpha_sample_only(flags: u32) -> bool {
    flags & AREA_LIGHT_FLAG_ZERO_ALPHA_SAMPLE_ONLY != 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    Point,
    Spot,
    Area,
    Distant,
    UniformInfinite,
    ImageInfinite,
    PortalImageInfinite,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Light {
    pub kind: LightKind,
    pub attributes: Vec<AttributeRef>,
    pub sampling_model: u32,
    pub image_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightGeometryKind {
    Position,
    Instance,
    Direction,
    Portal,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LightSamplingModel {
    pub kind: LightKind,
    pub geometry_kind: LightGeometryKind,
    pub geometry_index: u32,
    pub direction_index: u32,
    pub distribution_offset: u32,
    pub distribution_count: u32,
    pub total_area: f32,
    pub flags: u32,
    pub world_to_light: [[f32; 4]; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleDistributionEntry {
    pub primitive: u32,
    pub cdf: f32,
    pub area: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimitiveDistributionMap {
    pub offsets: Vec<u32>,
    pub entries: Vec<u32>,
}
