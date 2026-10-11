use crate::gpu::flat;
use crate::util::error::PbrtError;

use super::super::abi::{
    inverse_affine, row_major_to_columns, MediumRecord, VolumeMediumRecord, INVALID_INDEX,
};

pub struct ConvertedMedia {
    pub records: Vec<MediumRecord>,
    pub volumes: Vec<VolumeMediumRecord>,
    pub volume_data: Vec<f32>,
}

pub fn convert_media(
    media: &[flat::Medium],
    spectrum_attribute_count: usize,
) -> Result<ConvertedMedia, PbrtError> {
    let mut result = ConvertedMedia {
        records: Vec::with_capacity(media.len()),
        volumes: Vec::new(),
        volume_data: Vec::new(),
    };
    for (index, medium) in media.iter().enumerate() {
        let rgb_grid = matches!(&medium.data, flat::MediumData::RgbGrid(_));
        if !rgb_grid
            && (medium.sigma_a as usize >= spectrum_attribute_count
                || medium.sigma_s as usize >= spectrum_attribute_count
                || medium.le as usize >= spectrum_attribute_count)
        {
            return Err(PbrtError::error(&format!(
                "Flat medium {index} references an invalid spectrum."
            )));
        }
        let (kind, volume_index) = match (&medium.kind[..], &medium.data) {
            ("homogeneous", flat::MediumData::Homogeneous) => (0, INVALID_INDEX),
            ("uniformgrid", flat::MediumData::UniformGrid(grid)) => {
                let volume_index = u32::try_from(result.volumes.len())
                    .map_err(|_| PbrtError::error("Volume medium record count exceeds u32."))?;
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
                result.volumes.push(VolumeMediumRecord {
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
                    coefficient_offset_count: [0; 4],
                    majorant_offset_count: [majorant_offset, majorant_count, 0, 0],
                    medium_from_world,
                    cloud_parameters: [0.0; 4],
                });
                (1, volume_index)
            }
            ("cloud", flat::MediumData::Cloud(cloud)) => {
                let volume_index = u32::try_from(result.volumes.len())
                    .map_err(|_| PbrtError::error("Volume medium record count exceeds u32."))?;
                let majorant_offset = checked_volume_offset(result.volume_data.len())?;
                let majorant_count = 1u32;
                majorant_offset
                    .checked_add(majorant_count)
                    .ok_or_else(|| PbrtError::error("Cloud majorant range overflows u32."))?;
                result.volume_data.push(1.0);
                let medium_from_world = inverse_affine(medium.transform, "Cloud medium")?;
                result.volumes.push(VolumeMediumRecord {
                    bounds_min: [
                        cloud.bounds_min[0],
                        cloud.bounds_min[1],
                        cloud.bounds_min[2],
                        0.0,
                    ],
                    bounds_max: [
                        cloud.bounds_max[0],
                        cloud.bounds_max[1],
                        cloud.bounds_max[2],
                        0.0,
                    ],
                    resolution: [1, 1, 1, 0],
                    majorant_resolution: [1, 1, 1, 0],
                    density_offset_count: [0; 4],
                    coefficient_offset_count: [0; 4],
                    majorant_offset_count: [majorant_offset, majorant_count, 0, 0],
                    medium_from_world,
                    cloud_parameters: [cloud.density, cloud.wispiness, cloud.frequency, 0.0],
                });
                (2, volume_index)
            }
            ("rgbgrid", flat::MediumData::RgbGrid(grid)) => {
                let volume_index = u32::try_from(result.volumes.len())
                    .map_err(|_| PbrtError::error("Volume medium record count exceeds u32."))?;
                if grid.resolution.contains(&0)
                    || grid.majorant_resolution.contains(&0)
                    || grid.sigma_a.is_none() && grid.sigma_s.is_none()
                {
                    return Err(PbrtError::error(
                        "RGB grid requires positive resolutions and at least one coefficient grid.",
                    ));
                }
                let expected_voxels = grid
                    .resolution
                    .iter()
                    .try_fold(1usize, |count, value| count.checked_mul(*value as usize))
                    .ok_or_else(|| PbrtError::error("RGB grid voxel count overflows usize."))?;
                let expected_majorants = grid
                    .majorant_resolution
                    .iter()
                    .try_fold(1usize, |count, value| count.checked_mul(*value as usize))
                    .ok_or_else(|| PbrtError::error("RGB grid majorant count overflows usize."))?;
                if grid.majorant.len() != expected_majorants {
                    return Err(PbrtError::error(
                        "RGB grid majorant count does not match its resolution.",
                    ));
                }
                if grid
                    .bounds_min
                    .iter()
                    .chain(&grid.bounds_max)
                    .any(|value| !value.is_finite())
                    || (0..3).any(|axis| grid.bounds_min[axis] >= grid.bounds_max[axis])
                {
                    return Err(PbrtError::error(
                        "RGB grid bounds must be finite and non-empty.",
                    ));
                }
                if !grid.scale.is_finite() || grid.scale < 0.0 {
                    return Err(PbrtError::error(
                        "RGB grid scale must be finite and non-negative.",
                    ));
                }
                if grid
                    .majorant
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                {
                    return Err(PbrtError::error(
                        "RGB grid majorants must be finite and non-negative.",
                    ));
                }
                let sigma_a = append_rgb_grid(
                    &mut result.volume_data,
                    grid.sigma_a.as_deref(),
                    expected_voxels,
                    "sigma_a",
                )?;
                let sigma_s = append_rgb_grid(
                    &mut result.volume_data,
                    grid.sigma_s.as_deref(),
                    expected_voxels,
                    "sigma_s",
                )?;
                let majorant_offset = checked_volume_offset(result.volume_data.len())?;
                let majorant_count = u32::try_from(grid.majorant.len())
                    .map_err(|_| PbrtError::error("RGB grid majorant count exceeds u32."))?;
                majorant_offset
                    .checked_add(majorant_count)
                    .ok_or_else(|| PbrtError::error("RGB grid majorant range overflows u32."))?;
                result.volume_data.extend_from_slice(&grid.majorant);
                let medium_from_world = inverse_affine(medium.transform, "RGB grid medium")?;
                result.volumes.push(VolumeMediumRecord {
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
                        super::color_space_id(grid.color_space),
                    ],
                    majorant_resolution: [
                        grid.majorant_resolution[0],
                        grid.majorant_resolution[1],
                        grid.majorant_resolution[2],
                        0,
                    ],
                    density_offset_count: sigma_a,
                    coefficient_offset_count: sigma_s,
                    majorant_offset_count: [majorant_offset, majorant_count, 0, 0],
                    medium_from_world,
                    cloud_parameters: [grid.scale, 0.0, 0.0, 0.0],
                });
                (3, volume_index)
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
            volume_index,
            padding: [0; 2],
            medium_to_world: row_major_to_columns(medium.transform),
        });
    }
    Ok(result)
}

fn append_rgb_grid(
    volume_data: &mut Vec<f32>,
    values: Option<&[[f32; 3]]>,
    expected_voxels: usize,
    coefficient_name: &str,
) -> Result<[u32; 4], PbrtError> {
    let Some(values) = values else {
        return Ok([INVALID_INDEX, 0, 0, 0]);
    };
    if values.len() != expected_voxels {
        return Err(PbrtError::error(&format!(
            "RGB grid {coefficient_name} count does not match its resolution."
        )));
    }
    let value_count = values
        .len()
        .checked_mul(3)
        .ok_or_else(|| PbrtError::error("RGB grid coefficient buffer size overflows usize."))?;
    let offset = checked_volume_offset(volume_data.len())?;
    let count = u32::try_from(values.len())
        .map_err(|_| PbrtError::error("RGB grid coefficient count exceeds u32."))?;
    let scalar_count = u32::try_from(value_count)
        .map_err(|_| PbrtError::error("RGB grid coefficient value count exceeds u32."))?;
    offset
        .checked_add(scalar_count)
        .ok_or_else(|| PbrtError::error("RGB grid coefficient range overflows u32."))?;
    for rgb in values {
        if rgb.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(PbrtError::error(&format!(
                "RGB grid {coefficient_name} values must be finite and non-negative."
            )));
        }
        volume_data.extend_from_slice(rgb);
    }
    Ok([offset, count, 0, 0])
}

fn checked_volume_offset(length: usize) -> Result<u32, PbrtError> {
    u32::try_from(length).map_err(|_| PbrtError::error("GPU volume buffer exceeds u32 offsets."))
}
