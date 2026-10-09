use crate::gpu::flat;
use crate::util::error::PbrtError;

use super::super::abi::{
    instance_orientation_flags, inverse_transpose_linear, row_major_to_columns, Instance,
    INSTANCE_ORIENTATION_FLAG_SHAPE_TRANSFORM_SWAPS_HANDEDNESS, INVALID_INDEX,
};
use super::light::validate_instance_area_lights;

pub fn convert_instances(
    flat: &flat::Scene,
    geometry_count: usize,
) -> Result<Vec<Instance>, PbrtError> {
    flat.instances
        .iter()
        .enumerate()
        .map(|(index, instance)| {
            if instance.geometry as usize >= geometry_count {
                return Err(PbrtError::error(&format!(
                    "Flat instance {index} references an invalid geometry."
                )));
            }
            for (side, medium_id) in [
                ("inside", instance.inside_medium),
                ("outside", instance.outside_medium),
            ] {
                if medium_id != INVALID_INDEX && medium_id as usize >= flat.media.len() {
                    return Err(PbrtError::error(&format!(
                        "Flat instance {index} references an invalid {side} medium."
                    )));
                }
            }
            if instance.material_root != INVALID_INDEX
                && instance.material_root as usize >= flat.materials.roots.len()
            {
                return Err(PbrtError::error(&format!(
                    "Flat instance {index} references an invalid material."
                )));
            }
            validate_instance_area_lights(index, instance, flat)?;
            let label = format!("Flat instance {index}");
            Ok(Instance {
                geometry: instance.geometry,
                material_root: instance.material_root,
                area_light: instance.area_light,
                orientation_flags: instance_orientation_flags(
                    instance.reverse_orientation,
                    flat::transform_swaps_handedness(instance.transform),
                ) | if instance.shape_transform_swaps_handedness {
                    INSTANCE_ORIENTATION_FLAG_SHAPE_TRANSFORM_SWAPS_HANDEDNESS
                } else {
                    0
                },
                medium_inside: instance.inside_medium,
                medium_outside: instance.outside_medium,
                padding: [0; 2],
                world_from_object: row_major_to_columns(instance.transform),
                normal_from_object: inverse_transpose_linear(instance.transform, &label)?,
            })
        })
        .collect::<Result<Vec<_>, _>>()
}
