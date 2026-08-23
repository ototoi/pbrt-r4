use super::OPNode;
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;
use std::collections::HashMap;

fn rename_checked(
    params: &mut ParameterDictionary,
    before: &str,
    after: &str,
) -> Result<(), PbrtError> {
    if !params.has_parameter(before) {
        return Ok(());
    }
    if params.has_parameter(after) {
        return Err(PbrtError::error(&format!(
            "Cannot upgrade parameter \"{before}\" to \"{after}\": both are present."
        )));
    }
    params.rename_parameter(before, after);
    Ok(())
}

fn upgrade_film(params: &mut ParameterDictionary) -> Result<(), PbrtError> {
    if !params.has_parameter("maxsampleluminance") {
        return Ok(());
    }
    if params.has_parameter("maxcomponentvalue") {
        return Err(PbrtError::error(
            "Film has both \"maxsampleluminance\" and \"maxcomponentvalue\".",
        ));
    }
    let values = params.get_floats("maxsampleluminance");
    if values.len() != 1 {
        return Err(PbrtError::error(
            "Film \"maxsampleluminance\" requires exactly one value.",
        ));
    }
    params.rename_parameter("maxsampleluminance", "maxcomponentvalue");
    Ok(())
}

fn rgb_is_constant(params: &ParameterDictionary, name: &str, value: f32) -> bool {
    let values = params.get_points(name);
    values.len() == 3
        && values
            .iter()
            .all(|item| (*item - value).abs() <= f32::EPSILON)
}

fn rgb_is_equal_channels(params: &ParameterDictionary, name: &str) -> bool {
    let values = params.get_points(name);
    values.len() == 3 && values[0] == values[1] && values[1] == values[2]
}

fn remove_legacy_parameter(params: &mut ParameterDictionary, name: &str) {
    params.remove_parameter(name);
}

fn upgrade_uber(params: &mut ParameterDictionary, name: &mut String) -> Result<(), PbrtError> {
    *name = "coateddiffuse".to_string();
    if rgb_is_constant(params, "Ks", 0.0) {
        *name = "diffuse".to_string();
        remove_legacy_parameter(params, "eta");
        remove_legacy_parameter(params, "roughness");
    }
    remove_legacy_parameter(params, "Ks");
    remove_legacy_parameter(params, "Kr");
    remove_legacy_parameter(params, "Kt");
    rename_checked(params, "Kd", "reflectance")?;

    if params.has_parameter("opacity") {
        if !params.get_strings("opacity").is_empty() || !rgb_is_constant(params, "opacity", 1.0) {
            return Err(PbrtError::error(
                "Non-opaque \"opacity\" in legacy \"uber\" material is not supported.",
            ));
        }
        remove_legacy_parameter(params, "opacity");
    }
    Ok(())
}

fn upgrade_material_index(name: &str, params: &mut ParameterDictionary) -> Result<(), PbrtError> {
    if name != "glass" && name != "uber" {
        return Ok(());
    }
    rename_checked(params, "index", "eta")
}

fn upgrade_material(name: &mut String, params: &mut ParameterDictionary) -> Result<(), PbrtError> {
    let original_name = name.clone();
    upgrade_material_index(&original_name, params)?;
    rename_checked(params, "bumpmap", "displacement")?;
    match name.as_str() {
        "matte" => {
            *name = "diffuse".to_string();
            rename_checked(params, "Kd", "reflectance")?;
            remove_legacy_parameter(params, "sigma");
        }
        "kdsubsurface" => {
            *name = "subsurface".to_string();
            rename_checked(params, "Kd", "reflectance")?;
        }
        "disney" => {
            *name = "diffuse".to_string();
            rename_checked(params, "color", "reflectance")?;
        }
        "hair" => rename_checked(params, "color", "reflectance")?,
        "substrate" => {
            *name = "coateddiffuse".to_string();
            if rgb_is_constant(params, "Ks", 1.0) {
                remove_legacy_parameter(params, "Ks");
            }
            rename_checked(params, "Kd", "reflectance")?;
        }
        "plastic" => {
            *name = "coateddiffuse".to_string();
            if rgb_is_constant(params, "Ks", 0.0) {
                *name = "diffuse".to_string();
                remove_legacy_parameter(params, "roughness");
                remove_legacy_parameter(params, "eta");
            }
            remove_legacy_parameter(params, "Ks");
            rename_checked(params, "Kd", "reflectance")?;
        }
        "uber" => upgrade_uber(params, name)?,
        "mix" => {
            if params.has_parameter("amount") {
                let rgb = params.get_points("amount");
                if !rgb.is_empty() {
                    if rgb.len() != 3 {
                        return Err(PbrtError::error(
                            "Legacy mix \"amount\" requires an RGB value.",
                        ));
                    }
                    let amount = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
                    params.remove_parameter("amount");
                    params.add_float("amount", amount);
                } else if !params.get_spectrums("amount").is_empty() {
                    return Err(PbrtError::error(
                        "Non-RGB mix \"amount\" cannot be upgraded to a scalar.",
                    ));
                }
            }
            let material1 = params.get_one_string("namedmaterial1", "");
            let material2 = params.get_one_string("namedmaterial2", "");
            if material1.is_empty() || material2.is_empty() {
                if params.get_strings("materials").len() >= 2 {
                    return Ok(());
                }
                return Err(PbrtError::error(
                    "Legacy mix material requires namedmaterial1 and namedmaterial2.",
                ));
            }
            params.remove_parameter("namedmaterial1");
            params.remove_parameter("namedmaterial2");
            params.add_strings("string materials", &[&material2, &material1]);
        }
        "glass" => {
            *name = "dielectric".to_string();
            remove_legacy_parameter(params, "Kr");
            remove_legacy_parameter(params, "Kt");
        }
        "metal" => {
            *name = "conductor".to_string();
            remove_legacy_parameter(params, "Kr");
        }
        "translucent" => {
            *name = "diffusetransmission".to_string();
            rename_checked(params, "Kd", "transmittance")?;
            remove_legacy_parameter(params, "reflect");
            remove_legacy_parameter(params, "transmit");
            remove_legacy_parameter(params, "Ks");
            remove_legacy_parameter(params, "roughness");
        }
        "mirror" => {
            *name = "conductor".to_string();
            if params.has_parameter("roughness") {
                return Err(PbrtError::error(
                    "Legacy \"mirror\" material cannot provide \"roughness\".",
                ));
            }
            params.add_float("roughness", 0.0);
            params.add_string("spectrum eta", "metal-Ag-eta");
            params.add_string("spectrum k", "metal-Ag-k");
            remove_legacy_parameter(params, "Kr");
        }
        "fourier" => {
            return Err(PbrtError::error(
                "Legacy \"fourier\" material is unsupported; use \"measured\".",
            ));
        }
        "none" | "" => *name = "interface".to_string(),
        _ => {}
    }
    Ok(())
}

fn upgrade_scale_parameter(params: &mut ParameterDictionary) -> Result<(), PbrtError> {
    let values = params.get_points("scale");
    if values.is_empty() {
        return Ok(());
    }
    if values.len() != 3 || values[0] != values[1] || values[1] != values[2] {
        return Err(PbrtError::error(
            "Legacy RGB \"scale\" must be a constant value to upgrade.",
        ));
    }
    params.remove_parameter("scale");
    params.add_float("scale", values[0]);
    Ok(())
}

fn upgrade_scale_texture(params: &mut ParameterDictionary) -> Result<(), PbrtError> {
    let mut rgb_scale = None;
    let mut texture_name = None;
    for old_name in ["tex1", "tex2"] {
        if !params.has_parameter(old_name) {
            continue;
        }
        let key_type = params
            .get_keys()
            .into_iter()
            .find_map(|key| {
                let mut parts = key.split_ascii_whitespace();
                let first = parts.next()?;
                let second = parts.next()?;
                (second == old_name).then(|| first.to_string())
            })
            .unwrap_or_default();
        if key_type == "rgb" || key_type == "color" {
            let values = params.get_points(old_name);
            if values.len() != 3 || values[0] != values[1] || values[1] != values[2] {
                return Err(PbrtError::error(
                    "Non-constant RGB value found in legacy scale texture.",
                ));
            }
            if rgb_scale.replace(values[0]).is_some() {
                return Err(PbrtError::error(
                    "Legacy scale texture has two RGB parameters.",
                ));
            }
            params.remove_parameter(old_name);
        } else if key_type == "texture" {
            if texture_name.replace(old_name).is_some() {
                return Err(PbrtError::error(
                    "Legacy scale texture has two texture parameters.",
                ));
            }
        } else {
            return Err(PbrtError::error(
                "Legacy scale texture has an unsupported parameter type.",
            ));
        }
    }
    if let Some(old_name) = texture_name {
        rename_checked(params, old_name, "tex")?;
    }
    if let Some(value) = rgb_scale {
        if params.has_parameter("scale") {
            return Err(PbrtError::error(
                "Legacy scale texture has multiple scale parameters.",
            ));
        }
        params.add_float("scale", value);
    }
    Ok(())
}

fn upgrade_blackbody(params: &mut ParameterDictionary, name: &str) -> Result<f32, PbrtError> {
    if !params.has_parameter(name) {
        return Ok(1.0);
    }
    let values = params.get_floats(name);
    if values.len() != 2 {
        return Err(PbrtError::error(&format!(
            "Legacy blackbody \"{name}\" requires temperature and scale."
        )));
    }
    params.replace_blackbody(name, values[0]);
    Ok(values[1])
}

fn upgrade_directive(op: &mut OPNode) -> Result<(), PbrtError> {
    let Some(args) = op.args.as_mut() else {
        return Ok(());
    };
    let mut name = args.get_one_string("arg1", "");
    let Some(params) = op
        .params
        .as_mut()
        .and_then(crate::parser::ParameterStorage::as_dictionary_mut)
    else {
        return Ok(());
    };
    match op.name.as_str() {
        "PixelFilter" => {
            rename_checked(params, "xwidth", "xradius")?;
            rename_checked(params, "ywidth", "yradius")?;
            if name == "gaussian" && params.has_parameter("alpha") {
                let alpha = params.get_one_float("alpha", 0.0);
                if alpha <= 0.0 {
                    return Err(PbrtError::error("Gaussian alpha must be positive."));
                }
                params.remove_parameter("alpha");
                params.add_float("sigma", 1.0 / (2.0 * alpha).sqrt());
            }
        }
        "Film" => {
            upgrade_film(params)?;
            if params.has_parameter("scale") {
                let scale = params.get_one_float("scale", 1.0);
                params.remove_parameter("scale");
                params.add_float("iso", 100.0 * scale);
            }
            if name == "image" {
                name = "rgb".to_string();
            }
        }
        "Sampler" => {
            name = match name.as_str() {
                "lowdiscrepancy" | "02sequence" => "paddedsobol".to_string(),
                "maxmindist" => "pmj02bn".to_string(),
                "random" => "independent".to_string(),
                _ => name,
            };
        }
        "Integrator" => {
            params.remove_parameter("rrthreshold");
            if name == "directlighting" {
                name = "path".to_string();
                params.add_int("maxdepth", 1);
            }
            if name == "sppm" {
                params.remove_parameter("imagewritefrequency");
                params.remove_parameter("numiterations");
            }
            if params.get_one_string("lightsamplestrategy", "") == "spatial" {
                params.replace_one_string("lightsamplestrategy", "bvh");
            }
        }
        "Camera" => {
            if name == "environment" {
                name = "spherical".to_string();
                params.add_string("string mapping", "equirectangular");
            } else if name == "realistic" {
                params.remove_parameter("simpleweighting");
            }
        }
        "MakeNamedMedium" if name == "heterogeneous" => {
            name = "uniformgrid".to_string();
        }
        "LightSource" | "AreaLightSource" => {
            upgrade_scale_parameter(params)?;
            if op.name == "AreaLightSource" && name == "area" {
                name = "diffuse".to_string();
            }
            params.remove_parameter("samples");
            params.remove_parameter("nsamples");
            if params.has_parameter("mapname") {
                if name == "infinite"
                    && (!params.get_points("L").is_empty() || !params.get_spectrums("L").is_empty())
                    && !rgb_is_equal_channels(params, "L")
                {
                    return Err(PbrtError::error(
                        "Non-constant \"L\" is unsupported with \"mapname\" for an infinite light.",
                    ));
                }
                if name == "projection"
                    && (!params.get_points("I").is_empty() || !params.get_spectrums("I").is_empty())
                    && !rgb_is_equal_channels(params, "I")
                {
                    return Err(PbrtError::error(
                        "Non-constant \"I\" is unsupported with \"mapname\" for a projection light.",
                    ));
                }
            }
            let light_blackbody = upgrade_blackbody(params, "I")? * upgrade_blackbody(params, "L")?;
            if (light_blackbody - 1.0).abs() > f32::EPSILON {
                let scale = params.get_one_float("scale", 1.0) * light_blackbody;
                params.remove_parameter("scale");
                params.add_float("scale", scale);
            }
            if params.has_parameter("mapname") {
                rename_checked(params, "mapname", "filename")?;
            }
        }
        "Texture" => {
            let texname = args.get_one_string("arg3", "");
            if texname == "scale" {
                if args.get_one_string("arg2", "") == "float" {
                    rename_checked(params, "tex1", "tex")?;
                    rename_checked(params, "tex2", "scale")?;
                    upgrade_scale_parameter(params)?;
                } else {
                    upgrade_scale_texture(params)?;
                }
            }
            if texname == "imagemap" || texname == "ptex" {
                if params.has_parameter("trilinear") {
                    let trilinear = params.get_one_bool("trilinear", false);
                    params.remove_parameter("trilinear");
                    params.add_string(
                        "string filter",
                        if trilinear { "trilinear" } else { "bilinear" },
                    );
                }
                if params.has_parameter("gamma") {
                    if !params.get_floats("gamma").is_empty() {
                        let gamma = params.get_one_float("gamma", 0.0);
                        params.remove_parameter("gamma");
                        params.add_string("string encoding", &format!("gamma {gamma}"));
                    } else {
                        let gamma = params.get_one_bool("gamma", false);
                        params.remove_parameter("gamma");
                        params.add_string("string encoding", if gamma { "sRGB" } else { "linear" });
                    }
                }
            }
            let tex_type = args.get_one_string("arg2", "");
            if tex_type == "color" {
                args.replace_one_string("arg2", "spectrum");
            }
        }
        "Shape" => {
            rename_checked(params, "Kd", "reflectance")?;
            if name == "loopsubdiv" {
                rename_checked(params, "nlevels", "levels")?;
            }
            if name == "trianglemesh" {
                let indices = params.get_ints("indices");
                let points = params.get_point3f_array("P");
                if indices == [0, 1, 2] && points.len() == 3 {
                    params.remove_parameter("indices");
                }
            }
            if name == "bilinearmesh" {
                let indices = params.get_ints("indices");
                let points = params.get_point3f_array("P");
                if indices == [0, 1, 2, 3] && points.len() == 4 {
                    params.remove_parameter("indices");
                }
            }
            if name == "trianglemesh" || name == "plymesh" {
                params.remove_parameter("discarddegenerateUVs");
                params.remove_parameter("shadowalpha");
            }
            if name == "trianglemesh" {
                let uv = if params.has_parameter("st") {
                    let values = params.get_point2f_array("st");
                    params.remove_parameter("st");
                    values
                } else {
                    let values = params.get_point2f_array("uv");
                    params.remove_parameter("uv");
                    values
                };
                for value in uv {
                    params.add_point2f("point2 uv", &value);
                }
            }
        }
        _ => {}
    }
    args.replace_one_string("arg1", &name);
    Ok(())
}

pub fn upgrade_opnodes(ops: &mut [OPNode]) -> Result<(), PbrtError> {
    let mut texture_names = HashMap::<String, String>::new();
    let mut object_names = HashMap::<String, String>::new();
    let mut texture_rename_count = 0usize;
    let mut object_rename_count = 0usize;
    for op in ops {
        if let Some(params) = op.params.take() {
            op.params = Some(crate::parser::ParameterStorage::Dictionary(
                params.into_dictionary(),
            ));
        }
        for (before, after) in &texture_names {
            if let Some(params) = op
                .params
                .as_mut()
                .and_then(crate::parser::ParameterStorage::as_dictionary_mut)
            {
                params.rename_texture_references(before, after);
            }
        }
        match op.name.as_str() {
            "Material" | "MakeNamedMaterial" => {
                let Some(args) = op.args.as_mut() else {
                    return Err(PbrtError::error(&format!(
                        "{} requires a material name.",
                        op.name
                    )));
                };
                let mut name = args.get_one_string("arg1", "");
                if name.is_empty() {
                    return Err(PbrtError::error(&format!(
                        "{} requires a material name.",
                        op.name
                    )));
                }
                if let Some(params) = op
                    .params
                    .as_mut()
                    .and_then(crate::parser::ParameterStorage::as_dictionary_mut)
                {
                    upgrade_material(&mut name, params)?;
                }
                args.replace_one_string("arg1", &name);
            }
            "PixelFilter" | "Film" | "Sampler" | "Integrator" | "Camera" | "MakeNamedMedium"
            | "LightSource" | "AreaLightSource" | "Texture" | "Shape" => {
                if op.name == "Texture" {
                    if let Some(args) = op.args.as_mut() {
                        let original = args.get_one_string("arg1", "");
                        if let Some(previous) = texture_names.get(&original).cloned() {
                            let renamed = format!("{original}-renamed-{texture_rename_count}");
                            texture_rename_count += 1;
                            texture_names.insert(original.clone(), renamed.clone());
                            args.replace_one_string("arg1", &renamed);
                            if let Some(params) = op
                                .params
                                .as_mut()
                                .and_then(crate::parser::ParameterStorage::as_dictionary_mut)
                            {
                                params.rename_texture_references(&original, &previous);
                            }
                        } else {
                            texture_names.insert(original.clone(), original);
                        }
                    }
                }
                upgrade_directive(op)?;
            }
            "TransformBegin" => op.name = "AttributeBegin".to_string(),
            "TransformEnd" => op.name = "AttributeEnd".to_string(),
            "ObjectBegin" => {
                if let Some(args) = op.args.as_mut() {
                    let original = args.get_one_string("arg1", "");
                    let renamed = if object_names.contains_key(&original) {
                        let renamed = format!("{original}-renamed-{object_rename_count}");
                        object_rename_count += 1;
                        renamed
                    } else {
                        original.clone()
                    };
                    object_names.insert(original, renamed.clone());
                    args.replace_one_string("arg1", &renamed);
                }
            }
            "ObjectInstance" => {
                if let Some(args) = op.args.as_mut() {
                    let original = args.get_one_string("arg1", "");
                    if let Some(renamed) = object_names.get(&original) {
                        args.replace_one_string("arg1", renamed);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
