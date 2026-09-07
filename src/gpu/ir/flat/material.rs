use super::AttributeRef;

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub kind: String,
    pub source_kind: String,
    pub scattering_model: u32,
    pub attributes: Vec<AttributeRef>,
}

pub const EVENT_REFLECTION: u32 = 1 << 0;
pub const EVENT_TRANSMISSION: u32 = 1 << 1;
pub const EVENT_DIFFUSE: u32 = 1 << 2;
pub const EVENT_GLOSSY: u32 = 1 << 3;
pub const EVENT_SPECULAR: u32 = 1 << 4;

#[derive(Clone, Debug, PartialEq)]
pub struct ScatteringModel {
    pub surface_root: u32,
    pub bssrdf_root: u32,
}

/// A resolved view of a scattering model for backend lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedScatteringModel {
    pub root_node: u32,
    pub root_kind: String,
    pub event_flags: u32,
    pub data_index: u32,
    pub child_nodes: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScatteringNode {
    pub kind: String,
    pub event_flags: u32,
    pub data_index: u32,
    pub child_offset: u32,
    pub child_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScatteringChildRefs {
    pub node_ids: Vec<u32>,
}
