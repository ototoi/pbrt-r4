use super::AttributeRef;

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub kind: String,
    pub source_kind: String,
    pub attributes: Vec<AttributeRef>,
    pub tree_size: u32,
}

pub const EVENT_REFLECTION: u32 = 1 << 0;
pub const EVENT_TRANSMISSION: u32 = 1 << 1;
pub const EVENT_DIFFUSE: u32 = 1 << 2;
pub const EVENT_GLOSSY: u32 = 1 << 3;
pub const EVENT_SPECULAR: u32 = 1 << 4;
