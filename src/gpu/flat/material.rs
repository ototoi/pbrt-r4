use super::{AttributeKind, AttributeRef, Scene};
use crate::util::error::PbrtError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialTreeLayout {
    pub node_offset: u32,
    pub node_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialTreeNode {
    pub kind: String,
    pub source_kind: String,
    pub attributes: Vec<AttributeRef>,
    pub parent: u32,
    pub parent_slot: u32,
    pub child0: u32,
    pub child1: u32,
}

pub fn max_attributes_eval_work_items_per_surface(scene: &Scene) -> Result<u32, PbrtError> {
    scene.instances.iter().try_fold(0, |maximum, instance| {
        let layout = scene
            .material_tree_layouts
            .get(instance.material_tree_layout as usize)
            .ok_or_else(|| PbrtError::error("Instance material tree layout is missing."))?;
        Ok(maximum.max(layout.node_count))
    })
}

pub fn max_texture_eval_results_per_surface(scene: &Scene) -> Result<u32, PbrtError> {
    scene.instances.iter().try_fold(0, |maximum, instance| {
        let layout = scene
            .material_tree_layouts
            .get(instance.material_tree_layout as usize)
            .ok_or_else(|| PbrtError::error("Instance material tree layout is missing."))?;
        let start = layout.node_offset as usize;
        let end = start
            .checked_add(layout.node_count as usize)
            .ok_or_else(|| PbrtError::error("Material tree layout range overflowed."))?;
        let count = scene
            .material_tree_nodes
            .get(start..end)
            .ok_or_else(|| PbrtError::error("Material tree layout is outside the node table."))?
            .iter()
            .map(|node| {
                node.attributes
                    .iter()
                    .filter(|a| a.kind == AttributeKind::Texture)
                    .count() as u32
            })
            .try_fold(0u32, |sum, count| {
                sum.checked_add(count)
                    .ok_or_else(|| PbrtError::error("Flat texture result count overflowed."))
            })?;
        Ok(maximum.max(count))
    })
}

pub const EVENT_REFLECTION: u32 = 1 << 0;
pub const EVENT_TRANSMISSION: u32 = 1 << 1;
pub const EVENT_DIFFUSE: u32 = 1 << 2;
pub const EVENT_GLOSSY: u32 = 1 << 3;
pub const EVENT_SPECULAR: u32 = 1 << 4;
