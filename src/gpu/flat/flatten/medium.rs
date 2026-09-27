use super::{multiply_transform, FlatBuilder, Medium, Transform, INVALID_INDEX};
use crate::gpu::node::Medium as NodeMedium;
use crate::media::get_medium_scattering_properties;
use crate::util::error::PbrtError;
use crate::util::spectrum::{
    spectrum_to_photometric, DenselySampledSpectrum, Spectrum, SpectrumType,
};

pub fn register_medium(
    medium: &NodeMedium,
    world_transform: &Transform,
    builder: &mut FlatBuilder,
) -> Result<u32, PbrtError> {
    if let Some(index) = builder
        .media
        .iter()
        .position(|registered| registered.name == medium.name)
    {
        return u32::try_from(index)
            .map_err(|_| PbrtError::error("Flat medium table exceeds u32."));
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

    let mut sigma_a_dense = DenselySampledSpectrum::from_spectrum(&sigma_a);
    sigma_a_dense.scale(scale);
    let mut sigma_s_dense = DenselySampledSpectrum::from_spectrum(&sigma_s);
    sigma_s_dense.scale(scale);
    let mut le_dense = DenselySampledSpectrum::from_spectrum(&le);
    le_dense.scale(le_scale);

    let sigma_a_index = builder
        .spectrum_table_builder
        .intern_dense(&sigma_a_dense, 0)?;
    let sigma_s_index = builder
        .spectrum_table_builder
        .intern_dense(&sigma_s_dense, 0)?;
    let le_index = builder.spectrum_table_builder.intern_dense(&le_dense, 0)?;

    let index = u32::try_from(builder.media.len())
        .map_err(|_| PbrtError::error("Flat medium table exceeds u32."))?;
    builder.media.push(Medium {
        name: medium.name.clone(),
        kind: medium.kind.clone(),
        sigma_a: sigma_a_index,
        sigma_s: sigma_s_index,
        le: le_index,
        g: g as f32,
        transform: multiply_transform(world_transform, &medium.transform.matrix),
    });
    Ok(index)
}

/// Resolves a `MediumInterface`/`medium` name to an index into
/// `builder.media`. An empty name means vacuum (`INVALID_INDEX`); any
/// other name must already have been registered by `register_medium`
/// (Node IR always attaches `Component::Medium` before the shapes/camera
/// that reference it, since media are collected on the root node, which
/// is visited first).
pub fn resolve_medium_name(name: &str, builder: &FlatBuilder) -> Result<u32, PbrtError> {
    if name.is_empty() {
        return Ok(INVALID_INDEX);
    }
    builder
        .media
        .iter()
        .position(|registered| registered.name == name)
        .map(|index| index as u32)
        .ok_or_else(|| PbrtError::error(&format!("Medium \"{name}\" is not defined.")))
}
