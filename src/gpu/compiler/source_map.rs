use super::super::ir::{Index, SourceId};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuSourceMap {
    pub locations: Box<[super::GpuSourceLocation]>,
    pub resources: Box<[GpuSourceEntry]>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum GpuResourceKind {
    Transform,
    Spectrum,
    Image,
    TextureMapping,
    FloatTexture,
    SpectrumTexture,
    Material,
    Light,
    Geometry,
    Primitive,
    InstanceDefinition,
    Instance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSourceEntry {
    pub kind: GpuResourceKind,
    pub index: Index,
    pub source: SourceId,
}
