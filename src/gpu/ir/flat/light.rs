use super::AttributeRef;

pub const INVALID_INDEX: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    Point,
    Area,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Light {
    pub kind: LightKind,
    pub attributes: Vec<AttributeRef>,
    pub sampling_model: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightGeometryKind {
    Position,
    Instance,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LightSamplingModel {
    pub kind: LightKind,
    pub geometry_kind: LightGeometryKind,
    pub geometry_index: u32,
    pub distribution_offset: u32,
    pub distribution_count: u32,
    pub total_area: f32,
    pub flags: u32,
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
