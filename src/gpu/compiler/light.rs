use super::{
    compile_rgb_spectrum, compile_transform, finite_parameter, light_source_location,
    GpuCompileError, GpuDiffuseAreaLight, GpuIndex, GpuLight, GpuPointLight, GpuSourceLocation,
    GpuSpectrumResource, GpuSpectrumTexture, GpuTransform, GpuUniformInfiniteLight,
    LightSceneEntity, SceneBuilder, TransformId,
};
use crate::gpu::ir::{LightId, SpectrumTextureId};
use crate::parser::scene_builder::AreaLightSceneEntity;

pub fn compile_light(
    _builder: &SceneBuilder,
    light: &LightSceneEntity,
    transforms: &mut Vec<GpuTransform>,
    spectra: &mut Vec<GpuSpectrumResource>,
    lights: &mut Vec<GpuLight>,
) -> Result<(), GpuCompileError> {
    let source = light_source_location(light);
    let transform_id = TransformId(transforms.len() as GpuIndex);
    transforms.push(compile_transform(&light.base.render_from_object, &source)?);
    let (spectrum_name, default_scale) = match light.base.base.name.as_str() {
        "point" => ("I", 1.0),
        "infinite" => ("L", 1.0),
        _ => {
            return Err(GpuCompileError::UnsupportedSceneFeature {
                feature: "non-point/non-infinite light",
                source,
            })
        }
    };
    let spectrum = compile_rgb_spectrum(
        &light.base.base.params,
        spectrum_name,
        [1.0, 1.0, 1.0],
        &source,
        spectra,
    )?;
    let scale = finite_parameter(&light.base.base.params, "scale", default_scale, &source)?;
    match light.base.base.name.as_str() {
        "point" => lights.push(GpuLight::Point(GpuPointLight {
            render_from_light: transform_id,
            intensity: spectrum,
            scale,
        })),
        "infinite" => lights.push(GpuLight::UniformInfinite(GpuUniformInfiniteLight {
            radiance: spectrum,
            scale,
        })),
        _ => unreachable!(),
    }
    Ok(())
}

pub fn compile_area_light(
    area: &AreaLightSceneEntity,
    spectra: &mut Vec<GpuSpectrumResource>,
    spectrum_textures: &mut Vec<GpuSpectrumTexture>,
    lights: &mut Vec<GpuLight>,
    source: &GpuSourceLocation,
) -> Result<LightId, GpuCompileError> {
    if area.base.name != "diffuse" {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "non-diffuse area light",
            source: source.clone(),
        });
    }
    if !area.base.params.get_one_filename("filename", "").is_empty() {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "area light image emission",
            source: source.clone(),
        });
    }
    if area.base.params.get_textures_ref("L").is_some() {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "textured area light emission",
            source: source.clone(),
        });
    }
    let emission = compile_rgb_spectrum(&area.base.params, "L", [1.0, 1.0, 1.0], source, spectra)?;
    let emission_texture = SpectrumTextureId(spectrum_textures.len() as GpuIndex);
    spectrum_textures.push(GpuSpectrumTexture::Constant { value: emission });
    let scale = finite_parameter(&area.base.params, "scale", 1.0, source)?;
    let light_id = LightId(lights.len() as GpuIndex);
    lights.push(GpuLight::DiffuseArea(GpuDiffuseAreaLight {
        emission: emission_texture,
        scale,
        two_sided: area.base.params.get_one_bool("twosided", false),
    }));
    Ok(light_id)
}
