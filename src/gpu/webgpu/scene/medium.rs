use crate::gpu::flat;
use crate::util::error::PbrtError;

pub fn validate_storage_array_size<T: bytemuck::Pod>(
    limits: &wgpu::Limits,
    count: usize,
    label: &str,
) -> Result<(), PbrtError> {
    let byte_size = count
        .max(1)
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| PbrtError::error(&format!("GPU {label} buffer size overflows usize.")))?;
    let binding_limit = limits.max_storage_buffer_binding_size as usize;
    if byte_size > binding_limit {
        return Err(PbrtError::error(&format!(
            "GPU {label} buffer requires {byte_size} bytes; device storage binding limit is {binding_limit} bytes."
        )));
    }
    if byte_size as u64 > limits.max_buffer_size {
        return Err(PbrtError::error(&format!(
            "GPU {label} buffer requires {byte_size} bytes; device buffer limit is {} bytes.",
            limits.max_buffer_size
        )));
    }
    Ok(())
}

use super::super::abi::{
    inverse_affine, row_major_to_columns, MediumRecord, SparseGridEntryRecord,
    SparseGridRootRecord, VolumeMediumRecord, INVALID_INDEX,
};

pub struct ConvertedMedia {
    pub records: Vec<MediumRecord>,
    pub volumes: Vec<VolumeMediumRecord>,
    pub volume_data: Vec<f32>,
    pub sparse_grid_roots: Vec<SparseGridRootRecord>,
    pub sparse_grid_entries: Vec<SparseGridEntryRecord>,
}

pub fn convert_media(
    media: &[flat::Medium],
    spectrum_attribute_count: usize,
) -> Result<ConvertedMedia, PbrtError> {
    let mut result = ConvertedMedia {
        records: Vec::with_capacity(media.len()),
        volumes: Vec::new(),
        volume_data: Vec::new(),
        sparse_grid_roots: Vec::new(),
        sparse_grid_entries: Vec::new(),
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
                    sparse_grid_ranges: [[0; 4]; 2],
                    sparse_grid_background: [0.0; 4],
                    grid_inverse_map: [[0.0; 4]; 3],
                    grid_translation: [0.0; 4],
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
                    sparse_grid_ranges: [[0; 4]; 2],
                    sparse_grid_background: [0.0; 4],
                    grid_inverse_map: [[0.0; 4]; 3],
                    grid_translation: [0.0; 4],
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
                    sparse_grid_ranges: [[0; 4]; 2],
                    sparse_grid_background: [0.0; 4],
                    grid_inverse_map: [[0.0; 4]; 3],
                    grid_translation: [0.0; 4],
                });
                (3, volume_index)
            }
            ("nanovdb", flat::MediumData::NanoVDB(grid)) => {
                let volume_index = u32::try_from(result.volumes.len())
                    .map_err(|_| PbrtError::error("Volume medium record count exceeds u32."))?;
                let volume = append_sparse_grid(&mut result, grid, medium.transform)?;
                result.volumes.push(volume);
                (4, volume_index)
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

fn append_sparse_grid(
    output: &mut ConvertedMedia,
    grid: &flat::NanoVDBMedium,
    medium_to_world: [f32; 16],
) -> Result<VolumeMediumRecord, PbrtError> {
    use flat::sparse_grid::{SPARSE_LEAF_SIZE, SPARSE_LOWER_SIZE, SPARSE_UPPER_SIZE};
    grid.density.validate()?;
    if grid.majorant_resolution != [64; 3]
        || grid.majorant.len() != 64usize.pow(3)
        || grid.majorant.iter().any(|v| !v.is_finite() || *v < 0.0)
        || grid
            .bounds_min
            .iter()
            .chain(&grid.bounds_max)
            .any(|v| !v.is_finite())
        || (0..3).any(|axis| grid.bounds_min[axis] >= grid.bounds_max[axis])
    {
        return Err(PbrtError::error("Malformed Flat NanoVDB payload."));
    }
    let root_offset = u32::try_from(output.sparse_grid_roots.len())
        .map_err(|_| PbrtError::error("Sparse grid root offset exceeds u32."))?;
    let root_count = u32::try_from(grid.density.roots.len())
        .map_err(|_| PbrtError::error("Sparse grid root count exceeds u32."))?;
    root_offset
        .checked_add(root_count)
        .ok_or_else(|| PbrtError::error("Sparse grid root range overflows u32."))?;
    let entry_offset = u32::try_from(output.sparse_grid_entries.len())
        .map_err(|_| PbrtError::error("Sparse grid entry offset exceeds u32."))?;
    let upper_entries = &grid.density.upper_entries;
    let lower_entries = &grid.density.lower_entries;
    let entry_count = upper_entries
        .len()
        .checked_add(lower_entries.len())
        .ok_or_else(|| PbrtError::error("Sparse grid entry count overflows usize."))?;
    let entry_count = u32::try_from(entry_count)
        .map_err(|_| PbrtError::error("Sparse grid entry count exceeds u32."))?;
    entry_offset
        .checked_add(entry_count)
        .ok_or_else(|| PbrtError::error("Sparse grid entry range overflows u32."))?;
    let lower_offset = usize::try_from(entry_offset)
        .ok()
        .and_then(|base| base.checked_add(upper_entries.len()))
        .and_then(|base| u32::try_from(base).ok())
        .ok_or_else(|| PbrtError::error("Sparse lower entry offset exceeds u32."))?;
    let leaf_offset = checked_volume_offset(output.volume_data.len())?;
    let leaf_count = u32::try_from(grid.density.leaf_values.len())
        .map_err(|_| PbrtError::error("Sparse grid leaf value count exceeds u32."))?;
    leaf_offset
        .checked_add(leaf_count)
        .ok_or_else(|| PbrtError::error("Sparse grid leaf range overflows u32."))?;
    let majorant_offset = leaf_offset
        .checked_add(leaf_count)
        .ok_or_else(|| PbrtError::error("Sparse grid majorant offset overflows u32."))?;
    let majorant_count = u32::try_from(grid.majorant.len())
        .map_err(|_| PbrtError::error("Sparse grid majorant count exceeds u32."))?;
    majorant_offset
        .checked_add(majorant_count)
        .ok_or_else(|| PbrtError::error("Sparse grid majorant range overflows u32."))?;

    let upper_base = entry_offset;
    let lower_base = lower_offset;
    let leaf_base = leaf_offset;
    let mut root_records = Vec::with_capacity(grid.density.roots.len());
    for root in &grid.density.roots {
        let child_offset = match root.entry.child_index {
            None => INVALID_INDEX,
            Some(index) => absolute_node_offset(upper_base, index, SPARSE_UPPER_SIZE)?,
        };
        root_records.push(SparseGridRootRecord {
            block: root.block,
            child_offset,
            value: root.entry.value,
            padding: [0; 3],
        });
    }
    let mut entry_records = Vec::with_capacity(entry_count as usize);
    for entry in upper_entries {
        entry_records.push(SparseGridEntryRecord {
            child_offset: entry.child_index.map_or(Ok(INVALID_INDEX), |index| {
                absolute_node_offset(lower_base, index, SPARSE_LOWER_SIZE)
            })?,
            value: entry.value,
        });
    }
    for entry in lower_entries {
        entry_records.push(SparseGridEntryRecord {
            child_offset: entry.child_index.map_or(Ok(INVALID_INDEX), |index| {
                absolute_node_offset(leaf_base, index, SPARSE_LEAF_SIZE)
            })?,
            value: entry.value,
        });
    }
    output.sparse_grid_roots.extend(root_records);
    output.sparse_grid_entries.extend(entry_records);
    output
        .volume_data
        .extend_from_slice(&grid.density.leaf_values);
    output.volume_data.extend_from_slice(&grid.majorant);

    let medium_from_world = inverse_affine(medium_to_world, "NanoVDB medium")?;
    Ok(VolumeMediumRecord {
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
        resolution: [0; 4],
        majorant_resolution: [64, 64, 64, 0],
        density_offset_count: [0; 4],
        coefficient_offset_count: [0; 4],
        majorant_offset_count: [majorant_offset, majorant_count, 0, 0],
        medium_from_world,
        cloud_parameters: [0.0; 4],
        sparse_grid_ranges: [
            [root_offset, root_count, entry_offset, entry_count],
            [leaf_offset, leaf_count, 0, 0],
        ],
        sparse_grid_background: [grid.density.background, 0.0, 0.0, 0.0],
        grid_inverse_map: std::array::from_fn(|row| {
            [
                grid.inverse_map_f[row][0],
                grid.inverse_map_f[row][1],
                grid.inverse_map_f[row][2],
                0.0,
            ]
        }),
        grid_translation: [
            grid.translation_f[0],
            grid.translation_f[1],
            grid.translation_f[2],
            0.0,
        ],
    })
}

fn absolute_node_offset(base: u32, index: u32, node_size: usize) -> Result<u32, PbrtError> {
    let index =
        usize::try_from(index).map_err(|_| PbrtError::error("Sparse node index exceeds usize."))?;
    let relative = index
        .checked_mul(node_size)
        .and_then(|offset| u32::try_from(offset).ok())
        .ok_or_else(|| PbrtError::error("Sparse node element offset exceeds u32."))?;
    let absolute = base
        .checked_add(relative)
        .ok_or_else(|| PbrtError::error("Sparse node absolute offset overflows u32."))?;
    if absolute == INVALID_INDEX {
        return Err(PbrtError::error(
            "Sparse node offset collides with INVALID_INDEX.",
        ));
    }
    Ok(absolute)
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
