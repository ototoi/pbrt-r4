use crate::gpu::flat;
use crate::util::error::PbrtError;

use super::super::abi::{
    inverse_affine, row_major_to_columns, MediumRecord, UniformGridMediumRecord, INVALID_INDEX,
};

pub struct ConvertedMedia {
    pub records: Vec<MediumRecord>,
    pub uniform_grids: Vec<UniformGridMediumRecord>,
    pub volume_data: Vec<f32>,
}

pub fn convert_media(
    media: &[flat::Medium],
    spectrum_attribute_count: usize,
) -> Result<ConvertedMedia, PbrtError> {
    let mut result = ConvertedMedia {
        records: Vec::with_capacity(media.len()),
        uniform_grids: Vec::new(),
        volume_data: Vec::new(),
    };
    for (index, medium) in media.iter().enumerate() {
        if medium.sigma_a as usize >= spectrum_attribute_count
            || medium.sigma_s as usize >= spectrum_attribute_count
            || medium.le as usize >= spectrum_attribute_count
        {
            return Err(PbrtError::error(&format!(
                "Flat medium {index} references an invalid spectrum."
            )));
        }
        let (kind, grid_index) = match (&medium.kind[..], &medium.data) {
            ("homogeneous", flat::MediumData::Homogeneous) => (0, INVALID_INDEX),
            ("uniformgrid", flat::MediumData::UniformGrid(grid)) => {
                let grid_index = u32::try_from(result.uniform_grids.len())
                    .map_err(|_| PbrtError::error("Uniform grid record count exceeds u32."))?;
                let density_offset = checked_volume_offset(result.volume_data.len())?;
                let density_count = u32::try_from(grid.density.len())
                    .map_err(|_| PbrtError::error("Uniform grid density count exceeds u32."))?;
                density_offset
                    .checked_add(density_count)
                    .ok_or_else(|| PbrtError::error("Uniform grid density range overflows u32."))?;
                result.volume_data.extend_from_slice(&grid.density);
                let majorant_offset = checked_volume_offset(result.volume_data.len())?;
                let majorant_count = u32::try_from(grid.majorant.len())
                    .map_err(|_| PbrtError::error("Uniform grid majorant count exceeds u32."))?;
                majorant_offset.checked_add(majorant_count).ok_or_else(|| {
                    PbrtError::error("Uniform grid majorant range overflows u32.")
                })?;
                result.volume_data.extend_from_slice(&grid.majorant);
                let medium_from_world = inverse_affine(medium.transform, "Uniform grid medium")?;
                result.uniform_grids.push(UniformGridMediumRecord {
                    bounds_min: [
                        grid.bounds_min[0],
                        grid.bounds_min[1],
                        grid.bounds_min[2],
                        0.0,
                    ],
                    bounds_max: [
                        grid.bounds_max[0],
                        grid.bounds_max[1],
                        grid.bounds_max[2],
                        0.0,
                    ],
                    resolution: [
                        grid.resolution[0],
                        grid.resolution[1],
                        grid.resolution[2],
                        0,
                    ],
                    majorant_resolution: [
                        grid.majorant_resolution[0],
                        grid.majorant_resolution[1],
                        grid.majorant_resolution[2],
                        0,
                    ],
                    density_offset_count: [density_offset, density_count, 0, 0],
                    majorant_offset_count: [majorant_offset, majorant_count, 0, 0],
                    medium_from_world,
                });
                (1, grid_index)
            }
            (kind, _) => {
                return Err(PbrtError::error(&format!(
                    "Flat medium \"{}\" has inconsistent or unsupported type/data \"{kind}\".",
                    medium.name
                )));
            }
        };
        result.records.push(MediumRecord {
            kind,
            sigma_a: medium.sigma_a,
            sigma_s: medium.sigma_s,
            le: medium.le,
            g: medium.g,
            grid_index,
            padding: [0; 2],
            medium_to_world: row_major_to_columns(medium.transform),
        });
    }
    Ok(result)
}

fn checked_volume_offset(length: usize) -> Result<u32, PbrtError> {
    u32::try_from(length).map_err(|_| PbrtError::error("GPU volume buffer exceeds u32 offsets."))
}
