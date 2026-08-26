use super::{
    build_mip_storage, compile_rgb_spectrum, compile_transform, encode_mip_storage,
    finite_parameter, gpu_color_encoding, invalid_parameter, light_source_location, raw_encoding,
    raw_linear_pixels, DiffuseAreaLight, GpuCompileError, GpuSourceLocation, ImageChannels,
    ImageFilter, ImageResource, ImageWrapMode, Index, Light, LightSceneEntity, PointLight,
    SceneBuilder, SpectrumResource, SpectrumTexture, TextureMapping, Transform, TransformId,
    UniformInfiniteLight,
};
use crate::gpu::ir::{ImageId, LightId, SpectrumTextureId, SpectrumType, TextureMappingId};
use crate::parser::scene_builder::path_resolver::make_absolute_path;
use crate::parser::scene_builder::AreaLightSceneEntity;
use crate::util::imageio::{read_raw_image_with_encoding, ColorEncoding};
use std::path::Path;

pub fn compile_light(
    _builder: &SceneBuilder,
    light: &LightSceneEntity,
    transforms: &mut Vec<Transform>,
    spectra: &mut Vec<SpectrumResource>,
    lights: &mut Vec<Light>,
) -> Result<(), GpuCompileError> {
    let source = light_source_location(light);
    let transform_id = TransformId(transforms.len() as Index);
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
        "point" => lights.push(Light::Point(PointLight {
            render_from_light: transform_id,
            intensity: spectrum,
            scale,
        })),
        "infinite" => lights.push(Light::UniformInfinite(UniformInfiniteLight {
            radiance: spectrum,
            scale,
        })),
        _ => unreachable!(),
    }
    Ok(())
}

pub fn compile_area_light(
    builder: &SceneBuilder,
    area: &AreaLightSceneEntity,
    spectra: &mut Vec<SpectrumResource>,
    spectrum_textures: &mut Vec<SpectrumTexture>,
    images: &mut Vec<ImageResource>,
    texture_mappings: &mut Vec<TextureMapping>,
    lights: &mut Vec<Light>,
    source: &GpuSourceLocation,
) -> Result<LightId, GpuCompileError> {
    if area.base.name != "diffuse" {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "non-diffuse area light",
            source: source.clone(),
        });
    }
    if area.base.params.get_floats_ref("power").is_some() {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "area light power normalization",
            source: source.clone(),
        });
    }
    if area.base.params.get_textures_ref("L").is_some() {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "textured area light emission",
            source: source.clone(),
        });
    }
    let params = make_absolute_path(&area.base.params, &builder.seen_work_dirs);
    let filename = params.get_one_filename("filename", "");
    let emission_texture = if filename.is_empty() {
        let emission = compile_rgb_spectrum(&params, "L", [1.0, 1.0, 1.0], source, spectra)?;
        let emission_texture = SpectrumTextureId(spectrum_textures.len() as Index);
        spectrum_textures.push(SpectrumTexture::Constant { value: emission });
        emission_texture
    } else {
        if params.get_points_ref("L").is_some()
            || params.get_spectrums_ref("L").is_some()
            || params.get_sampled_spectra_ref("L").is_some()
        {
            return Err(invalid_parameter(
                "filename",
                "both L and filename are specified for a diffuse area light",
                source,
            ));
        }
        compile_area_light_image(
            &filename,
            images,
            texture_mappings,
            spectrum_textures,
            source,
        )?
    };
    let scale = finite_parameter(&area.base.params, "scale", 1.0, source)?;
    let light_id = LightId(lights.len() as Index);
    lights.push(Light::DiffuseArea(DiffuseAreaLight {
        emission: emission_texture,
        scale,
        two_sided: area.base.params.get_one_bool("twosided", false),
    }));
    Ok(light_id)
}

fn compile_area_light_image(
    filename: &str,
    images: &mut Vec<ImageResource>,
    texture_mappings: &mut Vec<TextureMapping>,
    spectrum_textures: &mut Vec<SpectrumTexture>,
    source: &GpuSourceLocation,
) -> Result<SpectrumTextureId, GpuCompileError> {
    let encoding = if Path::new(filename)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
    {
        ColorEncoding::parse("sRGB")
    } else {
        ColorEncoding::parse("linear")
    }
    .map_err(|error| GpuCompileError::InvalidParameter {
        parameter: "filename",
        detail: error.msg,
        source: source.clone(),
    })?;
    let raw = read_raw_image_with_encoding(filename, encoding).map_err(|error| {
        GpuCompileError::InvalidParameter {
            parameter: "filename",
            detail: error.msg,
            source: source.clone(),
        }
    })?;
    let width = u32::try_from(raw.resolution.x)
        .map_err(|_| invalid_parameter("filename", "image width must be positive", source))?;
    let height = u32::try_from(raw.resolution.y)
        .map_err(|_| invalid_parameter("filename", "image height must be positive", source))?;
    if width == 0 || height == 0 || !matches!(raw.channels, 3 | 4) {
        return Err(invalid_parameter(
            "filename",
            "diffuse area light image must have RGB or RGBA channels",
            source,
        ));
    }
    let source_pixels = raw_linear_pixels(&raw, source)?;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for pixel in source_pixels.chunks_exact(raw.channels) {
        pixels.extend_from_slice(&pixel[..3]);
    }
    let (linear_storage, mip_levels) = build_mip_storage(&pixels, width, height, 3);
    let encoding = raw_encoding(&raw);
    let storage = encode_mip_storage(&linear_storage, &raw.data, encoding, 3);
    let image = ImageId(images.len() as Index);
    images.push(ImageResource {
        resolution: [width, height],
        channels: ImageChannels::Rgb,
        storage,
        mip_levels,
        color_encoding: gpu_color_encoding(encoding),
    });
    let mapping = TextureMappingId(texture_mappings.len() as Index);
    texture_mappings.push(TextureMapping::Uv {
        su: 1.0,
        sv: -1.0,
        du: 0.0,
        dv: 1.0,
    });
    let texture = SpectrumTextureId(spectrum_textures.len() as Index);
    spectrum_textures.push(SpectrumTexture::Image {
        image,
        mapping,
        scale: 1.0,
        invert: false,
        swrap: ImageWrapMode::Clamp,
        twrap: ImageWrapMode::Clamp,
        filter: ImageFilter::Bilinear,
        spectrum_type: SpectrumType::Illuminant,
    });
    Ok(texture)
}
