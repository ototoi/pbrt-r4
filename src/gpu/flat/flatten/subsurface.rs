use super::material::texture_attribute_ref_with_spectrum_type;
use super::{push_spectrum_attribute, FlatBuilder};
use crate::gpu::flat::{AttributeRef, BSSRDFCoefficientKind, TabulatedBSSRDFTable, BSSRDF};
use crate::gpu::node::Material as NodeMaterial;
use crate::media::get_medium_scattering_properties;
use crate::util::error::PbrtError;
use crate::util::spectrum::{Spectrum, SpectrumType};

// pbrt-v4 SubsurfaceMaterial::Create: preset, sigma pair, reflectance/mfp, defaults.
pub fn register_subsurface(
    material: &NodeMaterial,
    builder: &mut FlatBuilder,
) -> Result<u32, PbrtError> {
    if !material.params.get_one_string("normalmap", "").is_empty() {
        return Err(PbrtError::error(&format!(
            "Material \"{}\": WebGPU subsurface normal maps are not supported yet.",
            material.name
        )));
    }
    let scale = material.params.get_one_float("scale", 1.0) as f32;
    let eta = material.params.get_one_float("eta", 1.33) as f32;
    let mut g = material.params.get_one_float("g", 0.0) as f32;
    let name = material.params.get_one_string("name", "");
    let (coefficient_kind, coefficients) = if !name.is_empty() {
        let (sigma_a, sigma_s) = get_medium_scattering_properties(&name).ok_or_else(|| {
            PbrtError::error(&format!(
                "Material \"{}\": named subsurface medium \"{name}\" not found.",
                material.name
            ))
        })?;
        if g != 0.0 {
            log::warn!("Non-zero g ignored with named scattering coefficients.");
        }
        g = 0.0;
        (
            BSSRDFCoefficientKind::Sigma,
            [
                push_spectrum_attribute(builder, "sigma_a", &sigma_a)?,
                push_spectrum_attribute(builder, "sigma_s", &sigma_s)?,
            ],
        )
    } else {
        let has_a = has_parameter(material, "sigma_a");
        let has_s = has_parameter(material, "sigma_s");
        if has_a != has_s {
            return Err(PbrtError::error(&format!(
                "Material \"{}\": sigma_a and sigma_s are required together.",
                material.name
            )));
        }
        if has_a {
            (
                BSSRDFCoefficientKind::Sigma,
                [
                    coefficient(
                        material,
                        "sigma_a",
                        &Spectrum::zero(),
                        SpectrumType::Unbounded,
                        builder,
                    )?,
                    coefficient(
                        material,
                        "sigma_s",
                        &Spectrum::zero(),
                        SpectrumType::Unbounded,
                        builder,
                    )?,
                ],
            )
        } else if has_parameter(material, "reflectance") {
            (
                BSSRDFCoefficientKind::ReflectanceMfp,
                [
                    coefficient(
                        material,
                        "reflectance",
                        &Spectrum::zero(),
                        SpectrumType::Albedo,
                        builder,
                    )?,
                    coefficient(
                        material,
                        "mfp",
                        &Spectrum::one(),
                        SpectrumType::Unbounded,
                        builder,
                    )?,
                ],
            )
        } else {
            (
                BSSRDFCoefficientKind::Sigma,
                [
                    push_spectrum_attribute(
                        builder,
                        "sigma_a",
                        &Spectrum::from([0.0011, 0.0024, 0.014]),
                    )?,
                    push_spectrum_attribute(
                        builder,
                        "sigma_s",
                        &Spectrum::from([2.55, 3.21, 3.77]),
                    )?,
                ],
            )
        }
    };
    if !scale.is_finite() || !g.is_finite() || !eta.is_finite() || eta <= 0.0 {
        return Err(PbrtError::error(&format!(
            "Material \"{}\" has invalid subsurface scale, g, or eta.",
            material.name
        )));
    }
    let table_index = if let Some(index) = builder
        .bssrdf_tables
        .iter()
        .position(|table| table.g == g && table.eta == eta)
    {
        u32::try_from(index)
            .map_err(|_| PbrtError::error("Flat BSSRDF table count exceeds u32."))?
    } else {
        let index = u32::try_from(builder.bssrdf_tables.len())
            .map_err(|_| PbrtError::error("Flat BSSRDF table count exceeds u32."))?;
        let table = TabulatedBSSRDFTable::new(g, eta);
        if table
            .profile
            .iter()
            .chain(&table.rho_eff)
            .chain(&table.profile_cdf)
            .any(|v| !v.is_finite())
        {
            return Err(PbrtError::error(&format!(
                "Material \"{}\" produced a non-finite BSSRDF table.",
                material.name
            )));
        }
        builder.bssrdf_tables.push(table);
        index
    };
    let index = u32::try_from(builder.bssrdfs.len())
        .map_err(|_| PbrtError::error("Flat BSSRDF record count exceeds u32."))?;
    builder.bssrdfs.push(BSSRDF {
        scale,
        g,
        eta,
        table_index,
        coefficient_kind,
        coefficients,
    });
    Ok(index)
}

fn has_parameter(material: &NodeMaterial, key: &str) -> bool {
    material
        .params
        .get_keys()
        .iter()
        .any(|stored| material.params.get_key_name(stored) == key)
        || material
            .texture_attributes
            .iter()
            .any(|(name, _)| name == key)
}

fn coefficient(
    material: &NodeMaterial,
    key: &str,
    default: &Spectrum,
    spectrum_type: SpectrumType,
    builder: &mut FlatBuilder,
) -> Result<AttributeRef, PbrtError> {
    let texture_type = match spectrum_type {
        SpectrumType::Albedo => 0,
        SpectrumType::Unbounded => 1,
        SpectrumType::Illuminant => 2,
    };
    if let Some(attribute) =
        texture_attribute_ref_with_spectrum_type(material, key, builder, texture_type)?
    {
        return Ok(attribute);
    }
    if material.params.get_keys().iter().any(|stored| {
        material.params.get_key_type(stored) == "texture"
            && material.params.get_key_name(stored) == key
    }) {
        return Err(PbrtError::error(&format!(
            "Material \"{}\" has an unresolved subsurface texture \"{key}\".",
            material.name
        )));
    }
    let value = material
        .params
        .get_one_spectrum_typed(key, default, spectrum_type);
    push_spectrum_attribute(builder, key, &value)
}
