use super::super::ir::{Index, SourceId};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceMap {
    pub locations: Box<[super::SourceLocation]>,
    pub resources: Box<[SourceEntry]>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ResourceKind {
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
pub struct SourceEntry {
    pub kind: ResourceKind,
    pub index: Index,
    pub source: SourceId,
}
