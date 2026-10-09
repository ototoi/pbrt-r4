use super::portal::{PortalDistributionTexel, PortalImageInfiniteLight};
use super::AttributeRef;
use super::{ImageInfiniteDistributionTexel, LightBVH, LightBounds};

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
    ImageInfinite,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageInfiniteSamplingRecord {
    pub distribution_offset: u32,
    pub row_cdf_offset: u32,
    pub resolution: [u32; 2],
    pub light_to_render: [[f32; 4]; 3],
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

#[derive(Clone, Debug, PartialEq)]
pub struct InfiniteLightResources {
    pub portal_records: Vec<PortalImageInfiniteLight>,
    pub portal_distribution: Vec<PortalDistributionTexel>,
    pub image_records: Vec<ImageInfiniteSamplingRecord>,
    pub image_distribution: Vec<ImageInfiniteDistributionTexel>,
    pub image_row_cdf: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LightResources {
    pub sampling_models: Vec<LightSamplingModel>,
    pub positions: Vec<[f32; 3]>,
    pub triangle_distributions: Vec<TriangleDistributionEntry>,
    pub lights: Vec<Light>,
    pub infinite_lights: Vec<Light>,
    pub bounds: Vec<LightBounds>,
    pub bvh: LightBVH,
    pub primitive_distribution_map: PrimitiveDistributionMap,
    pub infinite_sampling: InfiniteLightResources,
}
