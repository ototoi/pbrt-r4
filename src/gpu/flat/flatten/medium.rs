use super::{multiply_transform, FlatBuilder, Medium, Transform, INVALID_INDEX};
use crate::gpu::node::Medium as NodeMedium;
use crate::media::get_medium_scattering_properties;
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
    if medium.kind != "homogeneous" {
        return Err(PbrtError::error(&format!(
            "GPU backend does not support medium type \"{}\" (medium \"{}\").",
            medium.kind, medium.name
        )));
    }

    let preset = medium.params.get_one_string("preset", "");
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
    let le_scale = medium.params.get_one_float("Lescale", 1.0)
        / if photometric > 0.0 { photometric } else { 1.0 };
    let g = medium.params.get_one_float("g", 0.0);
    let g_gpu = g as f32;

    let mut sigma_a_dense = DenselySampledSpectrum::from_spectrum(&sigma_a);
    let mut sigma_s_dense = DenselySampledSpectrum::from_spectrum(&sigma_s);
    if (0..crate::gpu::flat::DENSE_SAMPLE_COUNT)
        .any(|index| sigma_a_dense[index] < 0.0 || sigma_s_dense[index] < 0.0)
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
    if (0..crate::gpu::flat::DENSE_SAMPLE_COUNT)
        .any(|index| sigma_a_dense[index] < 0.0 || sigma_s_dense[index] < 0.0)
    {
        return Err(PbrtError::error(&format!(
            "GPU medium \"{}\" has negative sigma_a or sigma_s after scale.",
            medium.name
        )));
    }
    if (0..crate::gpu::flat::DENSE_SAMPLE_COUNT).any(|index| le_dense[index] != 0.0) {
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
