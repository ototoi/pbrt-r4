pub const INVALID_INDEX: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    Point,
    Area,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightRecord {
    pub kind: LightKind,
    pub payload: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PointLight {
    pub position: [f32; 3],
    pub intensity: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct AreaLight {
    pub instance: u32,
    pub distribution: TriangleDistributionRange,
    pub emission: [f32; 3],
    pub two_sided: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleDistributionRange {
    pub offset: u32,
    pub count: u32,
    pub total_area: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleDistributionEntry {
    pub primitive: u32,
    pub cdf: f32,
    pub area: f32,
}
