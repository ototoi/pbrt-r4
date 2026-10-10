use super::super::{MediumData, UniformGridMedium, DENSE_SAMPLE_COUNT};
use super::{multiply_transform, FlatBuilder, Medium, Transform, INVALID_INDEX};
use crate::gpu::node::Medium as NodeMedium;
use crate::media::get_medium_scattering_properties;
use crate::util::base::Point3f;
use crate::util::error::PbrtError;
use crate::util::spectrum::{
    spectrum_to_photometric, DenselySampledSpectrum, Spectrum, SpectrumType,
};
use std::sync::Arc;

pub fn register_medium(
    medium: &Arc<NodeMedium>,
    world_transform: &Transform,
    builder: &mut FlatBuilder,
) -> Result<u32, PbrtError> {
    let medium_key = Arc::as_ptr(medium);
    if let Some(index) = builder.medium.media_indices_by_node.get(&medium_key) {
        return Ok(*index);
    }
    if medium.kind != "homogeneous" && medium.kind != "uniformgrid" {
        return Err(PbrtError::error(&format!(
            "GPU backend does not support medium type \"{}\" (medium \"{}\").",
            medium.kind, medium.name
        )));
    }

    let preset = if medium.kind == "homogeneous" {
        medium.params.get_one_string("preset", "")
    } else {
        String::new()
    };
    let (sigma_a_default, sigma_s_default) = get_medium_scattering_properties(&preset)
        .unwrap_or_else(|| {
            if !preset.is_empty() {
                log::warn!("Medium preset \"{preset}\" not found.");
            }
            (Spectrum::one(), Spectrum::one())
        });
    let sigma_a =
        medium
            .params
            .get_one_spectrum_typed("sigma_a", &sigma_a_default, SpectrumType::Unbounded);
    let sigma_s =
        medium
            .params
            .get_one_spectrum_typed("sigma_s", &sigma_s_default, SpectrumType::Unbounded);
    let scale = medium.params.get_one_float("scale", 1.0);

    let le =
        medium
            .params
            .get_one_spectrum_typed("Le", &Spectrum::zero(), SpectrumType::Illuminant);
    let photometric = spectrum_to_photometric(&le);
    let le_scale = if medium.kind == "homogeneous" {
        medium.params.get_one_float("Lescale", 1.0)
    } else {
        1.0
    } / if photometric > 0.0 { photometric } else { 1.0 };
    let g = medium.params.get_one_float("g", 0.0);
    let g_gpu = g as f32;

    let mut sigma_a_dense = DenselySampledSpectrum::from_spectrum(&sigma_a);
    let mut sigma_s_dense = DenselySampledSpectrum::from_spectrum(&sigma_s);
    if (0..DENSE_SAMPLE_COUNT).any(|index| sigma_a_dense[index] < 0.0 || sigma_s_dense[index] < 0.0)
    {
        return Err(PbrtError::error(&format!(
            "GPU medium \"{}\" requires non-negative sigma_a and sigma_s.",
            medium.name
        )));
    }
    sigma_a_dense.scale(scale);
    sigma_s_dense.scale(scale);
    let mut le_dense = DenselySampledSpectrum::from_spectrum(&le);
    le_dense.scale(le_scale);

    if !scale.is_finite()
        || !g.is_finite()
        || !g_gpu.is_finite()
        || !sigma_a_dense.is_valid()
        || !sigma_s_dense.is_valid()
        || !le_dense.is_valid()
    {
        return Err(PbrtError::error(&format!(
            "GPU medium \"{}\" contains non-finite coefficients.",
            medium.name
        )));
    }
    if scale < 0.0 {
        return Err(PbrtError::error(&format!(
            "GPU medium \"{}\" requires a non-negative scale.",
            medium.name
        )));
    }
    if (0..DENSE_SAMPLE_COUNT).any(|index| sigma_a_dense[index] < 0.0 || sigma_s_dense[index] < 0.0)
    {
        return Err(PbrtError::error(&format!(
            "GPU medium \"{}\" has negative sigma_a or sigma_s after scale.",
            medium.name
        )));
    }
    if (0..DENSE_SAMPLE_COUNT).any(|index| le_dense[index] != 0.0) {
        return Err(PbrtError::error(&format!(
            "GPU medium \"{}\" does not support nonzero Le.",
            medium.name
        )));
    }

    let sigma_a_index = builder
        .spectrum_table_builder
        .intern_dense(&sigma_a_dense, 0)?;
    let sigma_s_index = builder
        .spectrum_table_builder
        .intern_dense(&sigma_s_dense, 0)?;
    let le_index = builder.spectrum_table_builder.intern_dense(&le_dense, 0)?;

    let index = u32::try_from(builder.medium.media.len())
        .map_err(|_| PbrtError::error("Flat medium table exceeds u32."))?;
    builder.medium.media.push(Medium {
        name: medium.name.clone(),
        kind: medium.kind.clone(),
        sigma_a: sigma_a_index,
        sigma_s: sigma_s_index,
        le: le_index,
        g: g_gpu,
        transform: multiply_transform(world_transform, &medium.transform.matrix),
        data: if medium.kind == "uniformgrid" {
            MediumData::UniformGrid(create_uniform_grid(medium)?)
        } else {
            MediumData::Homogeneous
        },
    });
    builder
        .medium
        .media_indices_by_node
        .insert(medium_key, index);
    builder.medium.medium_refs.push(Arc::clone(medium));
    Ok(index)
}

/// Returns the flat table index of a Node IR medium reference. Vacuum maps to
/// `INVALID_INDEX`; every other medium must have been registered from Scene.
pub fn resolve_medium_reference(
    medium: &Option<Arc<NodeMedium>>,
    builder: &FlatBuilder,
) -> Result<u32, PbrtError> {
    let Some(medium) = medium else {
        return Ok(INVALID_INDEX);
    };
    builder
        .medium
        .media_indices_by_node
        .get(&Arc::as_ptr(medium))
        .copied()
        .ok_or_else(|| {
            PbrtError::error(&format!(
                "Medium \"{}\" is not in Scene.media.",
                medium.name
            ))
        })
}

fn create_uniform_grid(medium: &NodeMedium) -> Result<UniformGridMedium, PbrtError> {
    let nx = medium.params.get_one_int("nx", 1);
    let ny = medium.params.get_one_int("ny", 1);
    let nz = medium.params.get_one_int("nz", 1);
    if nx <= 0 || ny <= 0 || nz <= 0 {
        return Err(PbrtError::error(&format!(
            "GPU uniformgrid medium \"{}\" requires positive nx, ny, and nz.",
            medium.name
        )));
    }
    let resolution = [nx as u32, ny as u32, nz as u32];
    let expected = resolution
        .iter()
        .try_fold(1usize, |count, value| count.checked_mul(*value as usize))
        .ok_or_else(|| PbrtError::error("GPU uniformgrid density size overflows usize."))?;
    let density_values = medium.params.get_floats_ref("density").ok_or_else(|| {
        PbrtError::error(&format!(
            "GPU uniformgrid medium \"{}\" requires density.",
            medium.name
        ))
    })?;
    if density_values.len() != expected {
        return Err(PbrtError::error(&format!(
            "GPU uniformgrid medium \"{}\" has {} density values; expected {expected}.",
            medium.name,
            density_values.len()
        )));
    }
    let density: Vec<f32> = density_values.iter().map(|value| *value as f32).collect();
    if density
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PbrtError::error(&format!(
            "GPU uniformgrid medium \"{}\" requires finite, non-negative density values.",
            medium.name
        )));
    }
    if medium
        .params
        .get_floats_ref("temperature")
        .is_some_and(|values| !values.is_empty())
    {
        return Err(PbrtError::error(&format!(
            "GPU uniformgrid medium \"{}\" does not support temperature emission yet.",
            medium.name
        )));
    }
    let lescale = medium.params.get_floats_ref("Lescale");
    if let Some(values) = lescale {
        if !values.is_empty() && values.len() != expected {
            return Err(PbrtError::error(&format!(
                "GPU uniformgrid medium \"{}\" has invalid Lescale grid length {}.",
                medium.name,
                values.len()
            )));
        }
    }

    let p0 = medium
        .params
        .get_one_point3f("p0", &Point3f::new(0.0, 0.0, 0.0));
    let p1 = medium
        .params
        .get_one_point3f("p1", &Point3f::new(1.0, 1.0, 1.0));
    let p0 = [p0.x as f32, p0.y as f32, p0.z as f32];
    let p1 = [p1.x as f32, p1.y as f32, p1.z as f32];
    if p0.iter().chain(&p1).any(|value| !value.is_finite()) {
        return Err(PbrtError::error(&format!(
            "GPU uniformgrid medium \"{}\" requires finite bounds.",
            medium.name
        )));
    }
    let bounds_min = std::array::from_fn(|axis| p0[axis].min(p1[axis]));
    let bounds_max = std::array::from_fn(|axis| p0[axis].max(p1[axis]));
    if (0..3).any(|axis| {
        bounds_min[axis] >= bounds_max[axis] || !(bounds_max[axis] - bounds_min[axis]).is_finite()
    }) {
        return Err(PbrtError::error(&format!(
            "GPU uniformgrid medium \"{}\" requires finite, non-empty bounds.",
            medium.name
        )));
    }

    let majorant_resolution = [16u32; 3];
    let mut majorant = vec![0.0f32; 16 * 16 * 16];
    for z in 0..16i32 {
        for y in 0..16i32 {
            for x in 0..16i32 {
                let cell = [x, y, z];
                let mut lo = [0i32; 3];
                let mut hi = [0i32; 3];
                for axis in 0..3 {
                    let n = resolution[axis] as i32;
                    let p0 = cell[axis] as f32 / 16.0 * n as f32 - 0.5;
                    let p1 = (cell[axis] + 1) as f32 / 16.0 * n as f32 - 0.5;
                    lo[axis] = (p0.floor() as i32).max(0);
                    hi[axis] = ((p1.floor() as i32) + 1).min(n - 1);
                }
                let mut max_value = 0.0f32;
                for iz in lo[2]..=hi[2] {
                    for iy in lo[1]..=hi[1] {
                        for ix in lo[0]..=hi[0] {
                            let index = ((iz as usize * resolution[1] as usize + iy as usize)
                                * resolution[0] as usize)
                                + ix as usize;
                            max_value = max_value.max(density[index]);
                        }
                    }
                }
                majorant[((z * 16 + y) * 16 + x) as usize] = max_value;
            }
        }
    }
    Ok(UniformGridMedium {
        bounds_min,
        bounds_max,
        resolution,
        density,
        majorant_resolution,
        majorant,
    })
}
