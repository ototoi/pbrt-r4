use super::{AttributeKind, AttributeRef, Scene};
use crate::util::error::PbrtError;

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub kind: String,
    pub source_kind: String,
    pub attributes: Vec<AttributeRef>,
}

pub fn max_attributes_eval_work_items_per_surface(scene: &Scene) -> Result<u32, PbrtError> {
    fn occurrence_count(
        material_index: u32,
        scene: &Scene,
        counts: &mut [Option<u32>],
        visiting: &mut [bool],
    ) -> Result<u32, PbrtError> {
        let index = usize::try_from(material_index)
            .map_err(|_| PbrtError::error("Flat material index does not fit usize."))?;
        let material = scene
            .materials
            .get(index)
            .ok_or_else(|| PbrtError::error("Flat material reference is outside the table."))?;
        if let Some(count) = counts[index] {
            return Ok(count);
        }
        if visiting[index] {
            return Err(PbrtError::error("Flat material graph contains a cycle."));
        }
        visiting[index] = true;
        let mut count = 1u32;
        for child in material
            .attributes
            .iter()
            .filter(|attribute| attribute.kind == AttributeKind::Material)
        {
            count = count
                .checked_add(occurrence_count(child.index, scene, counts, visiting)?)
                .ok_or_else(|| PbrtError::error("Flat material work item count overflowed."))?;
        }
        visiting[index] = false;
        counts[index] = Some(count);
        Ok(count)
    }

    let mut counts = vec![None; scene.materials.len()];
    let mut visiting = vec![false; scene.materials.len()];
    scene.instances.iter().try_fold(0, |maximum, instance| {
        Ok(maximum.max(occurrence_count(
            instance.material,
            scene,
            &mut counts,
            &mut visiting,
        )?))
    })
}

pub const EVENT_REFLECTION: u32 = 1 << 0;
pub const EVENT_TRANSMISSION: u32 = 1 << 1;
pub const EVENT_DIFFUSE: u32 = 1 << 2;
pub const EVENT_GLOSSY: u32 = 1 << 3;
pub const EVENT_SPECULAR: u32 = 1 << 4;
