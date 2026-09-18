use super::material::{
    diffuse_reflectance, reject_scalar_textures, spectrum_attribute, texture_attribute_ref,
    texture_attribute_ref_unbounded,
};
use super::{
    push_scalar_attribute, push_spectrum_attribute, validate_layer_limits, AttributeRef,
    FlatBuilder,
};
use crate::gpu::node::Material as NodeMaterial;
use crate::util::error::PbrtError;
use crate::util::spectrum::{lookup_named_spectrum, Spectrum, SpectrumType};

pub fn build_material_attributes(
    source_material: &NodeMaterial,
    kind: &str,
    builder: &mut FlatBuilder,
) -> Result<Vec<AttributeRef>, PbrtError> {
    match kind {
        "mix" => {
            if let Some(attribute) = texture_attribute_ref(source_material, "amount", builder)? {
                Ok(vec![attribute])
            } else {
                let amount = source_material.params.get_one_float("amount", 0.5) as f32;
                if !amount.is_finite() {
                    return Err(PbrtError::error(&format!(
                        "Material \"{}\" has invalid mix amount.",
                        source_material.name
                    )));
                }
                Ok(vec![push_scalar_attribute(builder, "amount", amount)?])
            }
        }
        "coateddiffuse" => {
            let thickness = source_material.params.get_one_float("thickness", 0.01) as f32;
            let g = source_material.params.get_one_float("g", 0.0) as f32;
            let max_depth_i = source_material.params.get_one_int("maxdepth", 10);
            let n_samples_i = source_material.params.get_one_int("nsamples", 1);
            validate_layer_limits(&source_material.name, max_depth_i, n_samples_i)?;
            let max_depth = max_depth_i as f32;
            let n_samples = n_samples_i as f32;
            if ![thickness, g, max_depth, n_samples]
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(PbrtError::error(&format!(
                    "Material \"{}\" has invalid coateddiffuse parameters.",
                    source_material.name
                )));
            }
            let reflectance_attribute = if let Some(attribute) =
                texture_attribute_ref(source_material, "reflectance", builder)?
            {
                attribute
            } else {
                let reflectance = diffuse_reflectance(source_material)?;
                push_spectrum_attribute(builder, "reflectance", &reflectance)?
            };
            let albedo_attribute = if let Some(attribute) =
                texture_attribute_ref(source_material, "albedo", builder)?
            {
                attribute
            } else {
                push_spectrum_attribute(builder, "albedo", &Spectrum::from(0.0))?
            };
            let thickness_attribute = texture_attribute_ref(source_material, "thickness", builder)?
                .unwrap_or(push_scalar_attribute(builder, "thickness", thickness)?);
            let g_attribute = texture_attribute_ref(source_material, "g", builder)?
                .unwrap_or(push_scalar_attribute(builder, "g", g)?);
            let eta = spectrum_attribute(
                source_material,
                "eta",
                &Spectrum::from(1.5),
                SpectrumType::Unbounded,
            )?;
            let roughness = source_material.params.get_one_float("roughness", 0.0);
            let u_roughness = source_material
                .params
                .get_one_float("uroughness", roughness) as f32;
            let v_roughness = source_material
                .params
                .get_one_float("vroughness", roughness) as f32;
            let u_roughness_attribute =
                texture_attribute_ref(source_material, "uroughness", builder)?
                    .or(texture_attribute_ref(
                        source_material,
                        "roughness",
                        builder,
                    )?)
                    .unwrap_or(push_scalar_attribute(builder, "uroughness", u_roughness)?);
            let v_roughness_attribute =
                texture_attribute_ref(source_material, "vroughness", builder)?
                    .or(texture_attribute_ref(
                        source_material,
                        "roughness",
                        builder,
                    )?)
                    .unwrap_or(push_scalar_attribute(builder, "vroughness", v_roughness)?);
            let remap = source_material.params.get_one_bool("remaproughness", true);
            Ok(vec![
                thickness_attribute,
                reflectance_attribute,
                g_attribute,
                push_scalar_attribute(builder, "maxdepth", max_depth)?,
                push_scalar_attribute(builder, "nsamples", n_samples)?,
                albedo_attribute,
                push_spectrum_attribute(builder, "eta", &eta)?,
                u_roughness_attribute,
                v_roughness_attribute,
                push_scalar_attribute(builder, "remaproughness", if remap { 1.0 } else { 0.0 })?,
            ])
        }
        "coatedconductor" => {
            let thickness = source_material.params.get_one_float("thickness", 0.01) as f32;
            let g = source_material.params.get_one_float("g", 0.0) as f32;
            let max_depth_i = source_material.params.get_one_int("maxdepth", 10);
            let n_samples_i = source_material.params.get_one_int("nsamples", 1);
            validate_layer_limits(&source_material.name, max_depth_i, n_samples_i)?;
            let max_depth = max_depth_i as f32;
            let n_samples = n_samples_i as f32;
            if ![thickness, g, max_depth, n_samples]
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(PbrtError::error(&format!(
                    "Material \"{}\" has invalid coatedconductor parameters.",
                    source_material.name
                )));
            }
            let thickness_attribute = texture_attribute_ref(source_material, "thickness", builder)?
                .unwrap_or(push_scalar_attribute(builder, "thickness", thickness)?);
            let g_attribute = texture_attribute_ref(source_material, "g", builder)?
                .unwrap_or(push_scalar_attribute(builder, "g", g)?);
            let albedo_attribute =
                texture_attribute_ref(source_material, "albedo", builder)?.unwrap_or(
                    push_spectrum_attribute(builder, "albedo", &Spectrum::from(0.0))?,
                );
            let interface_eta = spectrum_attribute(
                source_material,
                "interface.eta",
                &Spectrum::from(1.5),
                SpectrumType::Unbounded,
            )?;
            let interface_eta_attribute =
                push_spectrum_attribute(builder, "interface.eta", &interface_eta)?;
            let interface_roughness = source_material
                .params
                .get_one_float("interface.roughness", 0.0);
            let interface_u = source_material
                .params
                .get_one_float("interface.uroughness", interface_roughness)
                as f32;
            let interface_v = source_material
                .params
                .get_one_float("interface.vroughness", interface_roughness)
                as f32;
            let interface_u_attribute =
                texture_attribute_ref(source_material, "interface.uroughness", builder)?
                    .or(texture_attribute_ref(
                        source_material,
                        "interface.roughness",
                        builder,
                    )?)
                    .unwrap_or(push_scalar_attribute(
                        builder,
                        "interface.uroughness",
                        interface_u,
                    )?);
            let interface_v_attribute =
                texture_attribute_ref(source_material, "interface.vroughness", builder)?
                    .or(texture_attribute_ref(
                        source_material,
                        "interface.roughness",
                        builder,
                    )?)
                    .unwrap_or(push_scalar_attribute(
                        builder,
                        "interface.vroughness",
                        interface_v,
                    )?);
            let has_reflectance = source_material
                .params
                .get_keys()
                .iter()
                .any(|key| source_material.params.get_key_name(key) == "reflectance");
            let conductor_eta_default = lookup_named_spectrum("metal-Cu-eta")
                .ok_or_else(|| PbrtError::error("Named spectrum metal-Cu-eta should exist."))?;
            let conductor_k_default = lookup_named_spectrum("metal-Cu-k")
                .ok_or_else(|| PbrtError::error("Named spectrum metal-Cu-k should exist."))?;
            let conductor_eta_attribute = if has_reflectance {
                push_spectrum_attribute(builder, "conductor.eta", &Spectrum::from(1.0))?
            } else if let Some(attribute) =
                texture_attribute_ref_unbounded(source_material, "conductor.eta", builder)?
            {
                attribute
            } else {
                let value = spectrum_attribute(
                    source_material,
                    "conductor.eta",
                    &conductor_eta_default,
                    SpectrumType::Unbounded,
                )?;
                push_spectrum_attribute(builder, "conductor.eta", &value)?
            };
            let conductor_k_attribute = if has_reflectance {
                push_spectrum_attribute(builder, "conductor.k", &Spectrum::from(0.0))?
            } else if let Some(attribute) =
                texture_attribute_ref_unbounded(source_material, "conductor.k", builder)?
            {
                attribute
            } else {
                let value = spectrum_attribute(
                    source_material,
                    "conductor.k",
                    &conductor_k_default,
                    SpectrumType::Unbounded,
                )?;
                push_spectrum_attribute(builder, "conductor.k", &value)?
            };
            let reflectance_attribute = if has_reflectance {
                if let Some(attribute) =
                    texture_attribute_ref(source_material, "reflectance", builder)?
                {
                    attribute
                } else {
                    let value = spectrum_attribute(
                        source_material,
                        "reflectance",
                        &Spectrum::from(0.5),
                        SpectrumType::Albedo,
                    )?;
                    push_spectrum_attribute(builder, "reflectance", &value)?
                }
            } else {
                push_spectrum_attribute(builder, "reflectance", &Spectrum::from(0.0))?
            };
            let conductor_roughness = source_material
                .params
                .get_one_float("conductor.roughness", 0.0);
            let conductor_u = source_material
                .params
                .get_one_float("conductor.uroughness", conductor_roughness)
                as f32;
            let conductor_v = source_material
                .params
                .get_one_float("conductor.vroughness", conductor_roughness)
                as f32;
            let conductor_u_attribute =
                texture_attribute_ref(source_material, "conductor.uroughness", builder)?
                    .or(texture_attribute_ref(
                        source_material,
                        "conductor.roughness",
                        builder,
                    )?)
                    .unwrap_or(push_scalar_attribute(
                        builder,
                        "conductor.uroughness",
                        conductor_u,
                    )?);
            let conductor_v_attribute =
                texture_attribute_ref(source_material, "conductor.vroughness", builder)?
                    .or(texture_attribute_ref(
                        source_material,
                        "conductor.roughness",
                        builder,
                    )?)
                    .unwrap_or(push_scalar_attribute(
                        builder,
                        "conductor.vroughness",
                        conductor_v,
                    )?);
            let remap = source_material.params.get_one_bool("remaproughness", true);
            Ok(vec![
                thickness_attribute,
                albedo_attribute,
                g_attribute,
                push_scalar_attribute(builder, "maxdepth", max_depth)?,
                push_scalar_attribute(builder, "nsamples", n_samples)?,
                interface_eta_attribute,
                interface_u_attribute,
                interface_v_attribute,
                conductor_eta_attribute,
                conductor_k_attribute,
                conductor_u_attribute,
                conductor_v_attribute,
                push_scalar_attribute(builder, "remaproughness", if remap { 1.0 } else { 0.0 })?,
                reflectance_attribute,
                push_scalar_attribute(
                    builder,
                    "use_reflectance",
                    if has_reflectance { 1.0 } else { 0.0 },
                )?,
            ])
        }
        "diffuse" => {
            if let Some(attribute) = texture_attribute_ref(source_material, "reflectance", builder)?
            {
                return Ok(vec![attribute]);
            }
            let reflectance = diffuse_reflectance(source_material)?;
            Ok(vec![push_spectrum_attribute(
                builder,
                "reflectance",
                &reflectance,
            )?])
        }
        "dielectric" | "thindielectric" => {
            let eta_attribute = if let Some(attribute) =
                texture_attribute_ref_unbounded(source_material, "eta", builder)?
            {
                attribute
            } else {
                let eta = spectrum_attribute(
                    source_material,
                    "eta",
                    &Spectrum::from(1.5),
                    SpectrumType::Unbounded,
                )?;
                let dense_eta = eta.to_dense();
                if (0..crate::util::spectrum::DENSE_SPECTRUM_SAMPLES)
                    .any(|index| !dense_eta[index].is_finite() || dense_eta[index] <= 0.0)
                {
                    return Err(PbrtError::error(&format!(
                        "Material \"{}\" has invalid dielectric eta.",
                        source_material.name
                    )));
                }
                push_spectrum_attribute(builder, "eta", &eta)?
            };
            if kind == "thindielectric" {
                return Ok(vec![eta_attribute]);
            }
            let roughness_value = source_material.params.get_one_float("roughness", 0.0);
            let u_roughness_value = source_material
                .params
                .get_one_float("uroughness", roughness_value);
            let v_roughness_value = source_material
                .params
                .get_one_float("vroughness", roughness_value);
            let u_roughness = u_roughness_value as f32;
            let v_roughness = v_roughness_value as f32;
            let u_roughness_attribute =
                texture_attribute_ref(source_material, "uroughness", builder)?
                    .or(texture_attribute_ref(
                        source_material,
                        "roughness",
                        builder,
                    )?)
                    .unwrap_or(push_scalar_attribute(builder, "uroughness", u_roughness)?);
            let v_roughness_attribute =
                texture_attribute_ref(source_material, "vroughness", builder)?
                    .or(texture_attribute_ref(
                        source_material,
                        "roughness",
                        builder,
                    )?)
                    .unwrap_or(push_scalar_attribute(builder, "vroughness", v_roughness)?);
            let remap = source_material.params.get_one_bool("remaproughness", true);
            Ok(vec![
                eta_attribute,
                u_roughness_attribute,
                v_roughness_attribute,
                push_scalar_attribute(builder, "remaproughness", if remap { 1.0 } else { 0.0 })?,
            ])
        }
        "conductor" => {
            reject_scalar_textures(source_material, &["uroughness", "vroughness"])?;
            let eta_attribute = if let Some(attribute) =
                texture_attribute_ref_unbounded(source_material, "eta", builder)?
            {
                attribute
            } else {
                let eta = spectrum_attribute(
                    source_material,
                    "eta",
                    &Spectrum::from(0.2),
                    SpectrumType::Unbounded,
                )?;
                let dense_eta = eta.to_dense();
                if (0..crate::util::spectrum::DENSE_SPECTRUM_SAMPLES)
                    .any(|index| !dense_eta[index].is_finite() || dense_eta[index] <= 0.0)
                {
                    return Err(PbrtError::error(&format!(
                        "Material \"{}\" has invalid conductor eta.",
                        source_material.name
                    )));
                }
                push_spectrum_attribute(builder, "eta", &eta)?
            };
            let k_attribute = if let Some(attribute) =
                texture_attribute_ref_unbounded(source_material, "k", builder)?
            {
                attribute
            } else {
                let k = spectrum_attribute(
                    source_material,
                    "k",
                    &Spectrum::from(3.0),
                    SpectrumType::Unbounded,
                )?;
                push_spectrum_attribute(builder, "k", &k)?
            };
            let roughness_attribute = if let Some(attribute) =
                texture_attribute_ref(source_material, "roughness", builder)?
            {
                attribute
            } else {
                let roughness = source_material.params.get_one_float("roughness", 0.0) as f32;
                if !roughness.is_finite() || roughness < 0.0 {
                    return Err(PbrtError::error(&format!(
                        "Material \"{}\" has invalid conductor roughness.",
                        source_material.name
                    )));
                }
                push_scalar_attribute(builder, "roughness", roughness)?
            };
            Ok(vec![eta_attribute, k_attribute, roughness_attribute])
        }
        _ => Err(PbrtError::error(&format!(
            "unsupported GPU material kind: {kind}"
        ))),
    }
}
