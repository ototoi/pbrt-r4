use super::{
    dot3, inverse_linear_transform, multiply_transform, push_scalar_attribute,
    push_spectrum_attribute, scale3, transform_point, transform_swaps_handedness, transform_vector,
    triangle_area, triangle_geometric_normal, AreaTriangleInput, FlatBuilder, Light,
    LightBoundInput, LightGeometryKind, LightKind, LightSamplingModel, Transform,
    TriangleDistributionEntry, IDENTITY_LINEAR_TRANSFORM, INVALID_INDEX,
};
use crate::gpu::node::{
    AreaLight as NodeAreaLight, Light as NodeLight, TextureComponent, TextureKind,
    TriangleMeshShape,
};
use crate::util::base::Point2f;
use crate::util::error::PbrtError;
use crate::util::geometry::equal_area_square_to_sphere;
use crate::util::spectrum::rgb_to_spectrum::{RGBColorSpace, ACES2065_1, DCI_P3, REC2020, SRGB};
use crate::util::spectrum::{spectrum_to_photometric, Spectrum, SpectrumType};

use super::super::portal::{prepare_portal_image, PortalImageInfiniteLight};
use super::super::texture::{build_linear_rgb_mipmap, ColorSpace, Mipmap, MipmapLevelData};

fn rgb_color_space(color_space: ColorSpace) -> &'static RGBColorSpace {
    match color_space {
        ColorSpace::Aces2065 => &ACES2065_1,
        ColorSpace::DciP3 => &DCI_P3,
        ColorSpace::Rec2020 => &REC2020,
        ColorSpace::Unknown | ColorSpace::Srgb => &SRGB,
    }
}

fn image_illuminant(color_space: ColorSpace) -> Spectrum {
    Spectrum::from(rgb_color_space(color_space).illuminant.to_dense())
}

fn image_illuminance(mipmap: &Mipmap) -> Result<f32, PbrtError> {
    let level = mipmap
        .levels
        .first()
        .ok_or_else(|| PbrtError::error("Image infinite light image has no mipmap levels."))?;
    let values = match &level.data {
        MipmapLevelData::F32(values) => values,
        _ => {
            return Err(PbrtError::error(
                "Image infinite light image must have linear float storage.",
            ))
        }
    };
    let luminance = rgb_color_space(mipmap.color_space).luminance_vector();
    let mut integral = 0.0f64;
    for y in 0..level.resolution[1] {
        let v = (y as f32 + 0.5) / level.resolution[1] as f32;
        for x in 0..level.resolution[0] {
            let u = (x as f32 + 0.5) / level.resolution[0] as f32;
            let w = equal_area_square_to_sphere(&Point2f::new(u, v));
            if w.z <= 0.0 {
                continue;
            }
            let offset = ((y * level.resolution[0] + x) * level.channels) as usize;
            let y_value = values[offset] * luminance[0]
                + values[offset + 1] * luminance[1]
                + values[offset + 2] * luminance[2];
            integral += (y_value * w.z) as f64;
        }
    }
    Ok((integral * 2.0 * std::f64::consts::PI
        / (level.resolution[0] as f64 * level.resolution[1] as f64)) as f32)
}

pub fn point_light(
    light: &NodeLight,
    parent_transform: &Transform,
    node_name: &str,
) -> Result<([f32; 3], Spectrum, f32, f32), PbrtError> {
    if light.name != "point" {
        return Err(PbrtError::error(&format!(
            "Unsupported GPU light \"{}\" on node \"{}\".",
            light.name, node_name
        )));
    }
    let from = light.params.get_one_point("from", &[0.0, 0.0, 0.0]);
    if from.len() != 3 || !from.iter().all(|value| value.is_finite()) {
        return Err(PbrtError::error(&format!(
            "Point light on node \"{}\" has an invalid from parameter.",
            node_name
        )));
    }
    let light_transform = multiply_transform(parent_transform, &light.transform.matrix);
    let position = transform_point(
        &light_transform,
        [from[0] as f32, from[1] as f32, from[2] as f32],
    );
    let white = Spectrum::from(1.0);
    let intensity = light
        .params
        .get_one_spectrum_typed("I", &white, SpectrumType::Illuminant);
    let mut scale = light.params.get_one_float("scale", 1.0);
    let photometric = spectrum_to_photometric(&intensity);
    if photometric > 0.0 {
        scale /= photometric;
    }
    let power = light.params.get_one_float("power", -1.0);
    if power > 0.0 {
        scale *= power / (4.0 * std::f32::consts::PI);
    }
    let intensity_max = intensity.max_value() as f32;
    if !position.iter().all(|value| value.is_finite()) || !scale.is_finite() {
        return Err(PbrtError::error(&format!(
            "Point light on node \"{}\" contains a non-finite value.",
            node_name
        )));
    }
    Ok((position, intensity, intensity_max, scale as f32))
}

pub fn spot_light(
    light: &NodeLight,
    parent_transform: &Transform,
    node_name: &str,
) -> Result<
    (
        [f32; 3],
        [f32; 3],
        Spectrum,
        f32,
        f32,
        f32,
        f32,
        [[f32; 4]; 3],
    ),
    PbrtError,
> {
    let from = light.params.get_one_point("from", &[0.0, 0.0, 0.0]);
    let to = light.params.get_one_point("to", &[0.0, 0.0, 1.0]);
    if from.len() != 3 || to.len() != 3 || !from.iter().chain(to.iter()).all(|v| v.is_finite()) {
        return Err(PbrtError::error(&format!(
            "Spot light on node \"{}\" has invalid from/to parameters.",
            node_name
        )));
    }
    let transform = multiply_transform(parent_transform, &light.transform.matrix);
    let position = transform_point(&transform, [from[0] as f32, from[1] as f32, from[2] as f32]);
    let direction = [
        to[0] as f32 - from[0] as f32,
        to[1] as f32 - from[1] as f32,
        to[2] as f32 - from[2] as f32,
    ];
    let length =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    let direction = if length > 0.0 {
        [
            direction[0] / length,
            direction[1] / length,
            direction[2] / length,
        ]
    } else {
        return Err(PbrtError::error(&format!(
            "Spot light on node \"{}\" has coincident from/to points.",
            node_name
        )));
    };
    let world_to_light = inverse_linear_transform(&transform).map_err(|message| {
        PbrtError::error(&format!(
            "Spot light on node \"{}\" has an invalid transform: {}.",
            node_name, message
        ))
    })?;
    let white = Spectrum::from(light.params.color_space().illuminant.to_dense());
    let intensity = light
        .params
        .get_one_spectrum_typed("I", &white, SpectrumType::Illuminant);
    let mut scale = light.params.get_one_float("scale", 1.0);
    let photometric = spectrum_to_photometric(&intensity);
    if photometric > 0.0 {
        scale /= photometric;
    }
    let cone = light.params.get_one_float("coneangle", 30.0);
    let delta = light.params.get_one_float("conedelta", 5.0);
    let delta = light.params.get_one_float("conedeltaangle", delta);
    let cos_end = (cone as f32).to_radians().cos();
    let cos_start = ((cone - delta) as f32).to_radians().cos();
    let power = light.params.get_one_float("power", -1.0);
    if power > 0.0 {
        let k_e = 2.0 * std::f32::consts::PI * ((1.0 - cos_start) + (cos_start - cos_end) / 2.0);
        scale *= power / k_e;
    }
    if !position
        .iter()
        .chain(direction.iter())
        .all(|v| v.is_finite())
        || !scale.is_finite()
    {
        return Err(PbrtError::error(&format!(
            "Spot light on node \"{}\" contains a non-finite value.",
            node_name
        )));
    }
    let intensity_max = intensity.max_value() as f32;
    Ok((
        position,
        direction,
        intensity,
        intensity_max,
        scale as f32,
        cos_start,
        cos_end,
        world_to_light,
    ))
}

pub fn distant_light(
    light: &NodeLight,
    parent_transform: &Transform,
    node_name: &str,
) -> Result<([f32; 3], Spectrum, f32), PbrtError> {
    let from = light.params.get_one_point("from", &[0.0, 0.0, 0.0]);
    let to = light.params.get_one_point("to", &[0.0, 0.0, 1.0]);
    if from.len() != 3 || to.len() != 3 || !from.iter().chain(to.iter()).all(|v| v.is_finite()) {
        return Err(PbrtError::error(&format!(
            "Distant light on node \"{}\" has invalid from/to parameters.",
            node_name
        )));
    }
    let transform = multiply_transform(parent_transform, &light.transform.matrix);
    let raw = transform_vector(
        &transform,
        [
            from[0] as f32 - to[0] as f32,
            from[1] as f32 - to[1] as f32,
            from[2] as f32 - to[2] as f32,
        ],
    );
    let length = dot3(raw, raw).sqrt();
    if length == 0.0 || !length.is_finite() {
        return Err(PbrtError::error(&format!(
            "Distant light on node \"{}\" has invalid direction.",
            node_name
        )));
    }
    let direction = [raw[0] / length, raw[1] / length, raw[2] / length];
    let white = Spectrum::from(light.params.color_space().illuminant.to_dense());
    let intensity = light
        .params
        .get_one_spectrum_typed("L", &white, SpectrumType::Illuminant);
    let photometric = spectrum_to_photometric(&intensity);
    let mut scale = light.params.get_one_float("scale", 1.0)
        / if photometric > 0.0 { photometric } else { 1.0 };
    let illuminance = light.params.get_one_float("illuminance", -1.0);
    if illuminance > 0.0 {
        scale *= illuminance;
    }
    Ok((direction, intensity, scale as f32))
}

pub fn infinite_light(
    light: &NodeLight,
    node_name: &str,
) -> Result<(LightKind, Spectrum, f32, f32), PbrtError> {
    let has_l = light.params.has_parameter("L");
    let filename = light.params.get_one_string("filename", "");
    let portal = light.params.get_points("portal");
    if portal.is_empty() && has_l && !filename.is_empty() {
        return Err(PbrtError::error(&format!(
            "Infinite light on node \"{node_name}\" cannot specify both L and filename without a portal."
        )));
    }
    if !portal.is_empty() && !has_l && filename.is_empty() {
        return Err(PbrtError::error(&format!(
            "Portal infinite light on node \"{node_name}\" requires L or filename."
        )));
    }
    let kind = if !portal.is_empty() {
        LightKind::PortalImageInfinite
    } else if !filename.is_empty() {
        LightKind::ImageInfinite
    } else {
        LightKind::UniformInfinite
    };
    let white = Spectrum::from(light.params.color_space().illuminant.to_dense());
    let intensity = light
        .params
        .get_one_spectrum_typed("L", &white, SpectrumType::Illuminant);
    let scale = light.params.get_one_float("scale", 1.0) as f32;
    let illuminance = light.params.get_one_float("illuminance", -1.0);
    if !scale.is_finite() {
        return Err(PbrtError::error(&format!(
            "Infinite light on node \"{}\" contains a non-finite scale.",
            node_name
        )));
    }
    Ok((kind, intensity, scale, illuminance as f32))
}

pub fn area_light_record(
    light: &NodeAreaLight,
    node_name: &str,
) -> Result<(Spectrum, f32, f32, bool), PbrtError> {
    if light.name != "diffuse" {
        return Err(PbrtError::error(&format!(
            "Unsupported GPU area light \"{}\" on node \"{}\".",
            light.name, node_name
        )));
    }
    if light.params.has_parameter("filename") {
        return Err(PbrtError::error(&format!(
            "Textured GPU area light on node \"{}\" is not implemented.",
            node_name
        )));
    }
    let white = Spectrum::from(1.0);
    let emission_spectrum =
        light
            .params
            .get_one_spectrum_typed("L", &white, SpectrumType::Illuminant);
    let photometric = spectrum_to_photometric(&emission_spectrum);
    let scale = light.params.get_one_float("scale", 1.0)
        / if photometric > 0.0 { photometric } else { 1.0 };
    let power = light.params.get_one_float("power", -1.0);
    if power > 0.0 {
        return Err(PbrtError::error(&format!(
            "GPU area light power on node \"{node_name}\" is not implemented."
        )));
    }
    let emission_max = emission_spectrum.max_value() as f32;
    if !scale.is_finite() {
        return Err(PbrtError::error(&format!(
            "GPU area light on node \"{}\" contains a non-finite emission value.",
            node_name
        )));
    }
    Ok((
        emission_spectrum,
        emission_max,
        scale as f32,
        light.params.get_one_bool("twosided", false),
    ))
}

pub fn flatten_light(
    light: NodeLight,
    world_transform: &Transform,
    name: &str,
    builder: &mut FlatBuilder,
) -> Result<(), PbrtError> {
    if light.name == "distant" {
        let (direction, intensity, scale) = distant_light(&light, &world_transform, &name)?;
        let direction_index = u32::try_from(builder.light_positions.len()).map_err(|_| {
            PbrtError::error("The flattened GPU light direction table exceeds u32.")
        })?;
        builder.light_positions.push(direction);
        let sampling_model = u32::try_from(builder.light_sampling_models.len()).map_err(|_| {
            PbrtError::error("The flattened GPU light sampling model table exceeds u32.")
        })?;
        builder.light_sampling_models.push(LightSamplingModel {
            kind: LightKind::Distant,
            geometry_kind: LightGeometryKind::Direction,
            geometry_index: direction_index,
            direction_index,
            distribution_offset: 0,
            distribution_count: 0,
            total_area: 0.0,
            flags: 0,
            world_to_light: IDENTITY_LINEAR_TRANSFORM,
        });
        let i_attr = push_spectrum_attribute(builder, "L", &intensity)?;
        let scale_attr = push_scalar_attribute(builder, "scale", scale)?;
        builder.infinite_lights.push(Light {
            kind: LightKind::Distant,
            attributes: vec![i_attr, scale_attr],
            sampling_model,
            image_index: INVALID_INDEX,
        });
    } else if light.name == "infinite" {
        let (kind, intensity, base_scale, illuminance) = infinite_light(&light, &name)?;
        let light_transform = multiply_transform(world_transform, &light.transform.matrix);
        let world_to_light = inverse_linear_transform(&light_transform).map_err(|message| {
            PbrtError::error(&format!(
                "Infinite light on node \"{}\" has an invalid transform: {}.",
                name, message
            ))
        })?;
        let (image_index, image_illuminant, map_illuminance, geometry_kind, geometry_index) = if matches!(
            kind,
            LightKind::ImageInfinite | LightKind::PortalImageInfinite
        ) {
            let filename = light.params.get_one_string("filename", "");
            if filename.is_empty() && kind == LightKind::ImageInfinite {
                return Err(PbrtError::error(
                    "Image infinite light is missing filename.",
                ));
            }
            let encoding = light.params.get_one_string("encoding", "srgb");
            let mipmap = if filename.is_empty() {
                let rgb = intensity.to_rgb();
                build_linear_rgb_mipmap(
                    [1, 1],
                    &[[rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]],
                    ColorSpace::Srgb,
                )?
            } else {
                builder
                    .infinite_light_image_decoder
                    .decode_linear_rgb(std::path::Path::new(&filename), &encoding)?
            };
            let base_level = mipmap.levels.first().ok_or_else(|| {
                PbrtError::error("Image infinite light image has no mipmap levels.")
            })?;
            if base_level.resolution[0] != base_level.resolution[1] {
                return Err(PbrtError::error(&format!(
                    "Image infinite light image \"{}\" has non-square resolution {:?}.",
                    filename, base_level.resolution
                )));
            }
            if base_level.channels < 3 {
                return Err(PbrtError::error(&format!(
                    "Image infinite light image \"{}\" does not have RGB channels.",
                    filename
                )));
            }
            let contains_non_finite = match &base_level.data {
                MipmapLevelData::F32(values) => values.iter().any(|value| !value.is_finite()),
                MipmapLevelData::F16(values) => values
                    .iter()
                    .any(|value| !half::f16::from_bits(*value).is_finite()),
                MipmapLevelData::U8(_) => false,
            };
            if contains_non_finite {
                return Err(PbrtError::error(&format!(
                    "Image infinite light image \"{}\" contains a non-finite value.",
                    filename
                )));
            }
            let illuminant = image_illuminant(mipmap.color_space);
            let map_illuminance = if illuminance > 0.0 {
                Some(image_illuminance(&mipmap)?)
            } else {
                None
            };
            let (mipmap, geometry_kind, geometry_index) = if kind == LightKind::PortalImageInfinite
            {
                let points = light.params.get_points("portal");
                if points.len() != 12 {
                    return Err(PbrtError::error(
                        "Portal image infinite light requires four portal points.",
                    ));
                }
                // Portal parameters have already been converted to render/world space by
                // Node IR construction. The light transform only orients the source map.
                let portal = std::array::from_fn(|i| {
                    [
                        points[3 * i] as f32,
                        points[3 * i + 1] as f32,
                        points[3 * i + 2] as f32,
                    ]
                });
                let prepared = prepare_portal_image(&mipmap, portal, world_to_light)?;
                let distribution_offset = u32::try_from(builder.portal_distribution.len())
                    .map_err(|_| PbrtError::error("Portal distribution offset exceeds u32."))?;
                builder
                    .portal_distribution
                    .extend_from_slice(&prepared.distribution);
                let geometry_index = u32::try_from(builder.portal_infinite_lights.len())
                    .map_err(|_| PbrtError::error("Portal image table exceeds u32."))?;
                builder
                    .portal_infinite_lights
                    .push(PortalImageInfiniteLight {
                        portal: prepared.portal,
                        world_to_portal: prepared.world_to_portal,
                        distribution_offset,
                        resolution: prepared.resolution,
                    });
                (prepared.mipmap, LightGeometryKind::Portal, geometry_index)
            } else {
                (mipmap, LightGeometryKind::Direction, INVALID_INDEX)
            };
            let index = u32::try_from(builder.infinite_light_mipmaps.len())
                .map_err(|_| PbrtError::error("Infinite light image table exceeds u32."))?;
            builder.infinite_light_mipmaps.push(mipmap);
            (
                index,
                Some(illuminant),
                map_illuminance,
                geometry_kind,
                geometry_index,
            )
        } else {
            (
                INVALID_INDEX,
                None,
                None,
                LightGeometryKind::Direction,
                INVALID_INDEX,
            )
        };
        let sampling_model = u32::try_from(builder.light_sampling_models.len()).map_err(|_| {
            PbrtError::error("The flattened GPU light sampling model table exceeds u32.")
        })?;
        builder.light_sampling_models.push(LightSamplingModel {
            kind,
            geometry_kind,
            geometry_index,
            direction_index: INVALID_INDEX,
            distribution_offset: 0,
            distribution_count: 0,
            total_area: 0.0,
            flags: 0,
            world_to_light: if kind == LightKind::PortalImageInfinite {
                IDENTITY_LINEAR_TRANSFORM
            } else {
                world_to_light
            },
        });
        let normalization = image_illuminant.as_ref().unwrap_or(&intensity);
        let photometric = spectrum_to_photometric(normalization);
        if !photometric.is_finite() || photometric <= 0.0 {
            return Err(PbrtError::error(&format!(
                "Infinite light on node \"{name}\" has non-positive photometric normalization."
            )));
        }
        let mut scale = base_scale / photometric as f32;
        if illuminance > 0.0 {
            let k_e = map_illuminance.unwrap_or(std::f32::consts::PI);
            if !k_e.is_finite() || k_e <= 0.0 {
                return Err(PbrtError::error(&format!(
                    "Infinite light on node \"{name}\" has non-positive illuminance normalization."
                )));
            }
            scale *= illuminance / k_e;
        }
        if !scale.is_finite() {
            return Err(PbrtError::error(&format!(
                "Infinite light on node \"{name}\" has a non-finite scale."
            )));
        }
        let i_attr = push_spectrum_attribute(builder, "L", &intensity)?;
        let scale_attr = push_scalar_attribute(builder, "scale", scale)?;
        let mut attributes = vec![i_attr, scale_attr];
        if let Some(illuminant) = image_illuminant {
            attributes.push(push_spectrum_attribute(
                builder,
                "image-illuminant",
                &illuminant,
            )?);
        }
        builder.infinite_lights.push(Light {
            kind,
            attributes,
            sampling_model,
            image_index,
        });
    } else {
        let (
            position,
            direction,
            intensity,
            intensity_max,
            scale,
            cos_start,
            cos_end,
            world_to_light,
        ) = match light.name.as_str() {
            "point" => {
                let (position, intensity, intensity_max, scale) =
                    point_light(&light, &world_transform, &name)?;
                (
                    position,
                    [0.0, 0.0, 0.0],
                    intensity,
                    intensity_max,
                    scale,
                    1.0,
                    -1.0,
                    IDENTITY_LINEAR_TRANSFORM,
                )
            }
            "spot" => spot_light(&light, &world_transform, &name)?,
            _ => {
                return Err(PbrtError::error(&format!(
                    "Unsupported GPU light \"{}\" on node \"{}\".",
                    light.name, name
                )))
            }
        };
        let position_index = u32::try_from(builder.light_positions.len())
            .map_err(|_| PbrtError::error("The flattened GPU light position table exceeds u32."))?;
        builder.light_positions.push(position);
        let direction_index = if light.name == "spot" {
            let index = u32::try_from(builder.light_positions.len()).map_err(|_| {
                PbrtError::error("The flattened GPU light direction table exceeds u32.")
            })?;
            builder.light_positions.push(direction);
            index
        } else {
            INVALID_INDEX
        };
        let sampling_model = u32::try_from(builder.light_sampling_models.len()).map_err(|_| {
            PbrtError::error("The flattened GPU light sampling model table exceeds u32.")
        })?;
        let kind = if light.name == "spot" {
            LightKind::Spot
        } else {
            LightKind::Point
        };
        builder.light_sampling_models.push(LightSamplingModel {
            kind,
            geometry_kind: LightGeometryKind::Position,
            geometry_index: position_index,
            direction_index,
            distribution_offset: 0,
            distribution_count: 0,
            total_area: 0.0,
            flags: 0,
            world_to_light,
        });
        builder.light_bound_inputs.push(LightBoundInput::Point {
            handle: u32::try_from(builder.lights.len())
                .map_err(|_| PbrtError::error("The flattened GPU light table exceeds u32."))?,
            world_position: position,
            intensity_max,
            scale,
        });
        let i_attr = push_spectrum_attribute(builder, "I", &intensity)?;
        let scale_attr = push_scalar_attribute(builder, "scale", scale)?;
        let mut attributes = vec![i_attr, scale_attr];
        if kind == LightKind::Spot {
            attributes.push(push_scalar_attribute(
                builder,
                "cos_falloff_start",
                cos_start,
            )?);
            attributes.push(push_scalar_attribute(builder, "cos_falloff_end", cos_end)?);
        }
        builder.lights.push(Light {
            kind,
            attributes,
            sampling_model,
            image_index: INVALID_INDEX,
        });
    }
    Ok(())
}
pub fn append_area_light(
    area_light: NodeAreaLight,
    shape: &TriangleMeshShape,
    name: &str,
    world_transform: &Transform,
    instance_index: u32,
    material_source: u32,
    reverse_orientation: bool,
    builder: &mut FlatBuilder,
) -> Result<u32, PbrtError> {
    let light_handle = u32::try_from(builder.lights.len())
        .map_err(|_| PbrtError::error("The flattened GPU light table exceeds u32."))?;
    let triangle_count = shape.indices.len() / 3;
    if triangle_count == 0 {
        return Err(PbrtError::error(&format!(
            "Area-light shape node \"{name}\" contains no triangles."
        )));
    }
    let (emission, emission_max, scale, two_sided) = area_light_record(&area_light, &name)?;
    let distribution_offset = u32::try_from(builder.triangle_distributions.len())
        .map_err(|_| PbrtError::error("The flattened GPU distribution table exceeds u32."))?;
    let mut total_area = 0.0;
    let mut bound_triangles = Vec::with_capacity(triangle_count);
    let mut entries = Vec::with_capacity(triangle_count);
    for primitive in 0..triangle_count {
        let primitive = u32::try_from(primitive)
            .map_err(|_| PbrtError::error("The flattened GPU area-light primitive exceeds u32."))?;
        let i0 = shape.indices[primitive as usize * 3] as usize;
        let i1 = shape.indices[primitive as usize * 3 + 1] as usize;
        let i2 = shape.indices[primitive as usize * 3 + 2] as usize;
        let positions = [
            transform_point(&world_transform, shape.positions[i0].0),
            transform_point(&world_transform, shape.positions[i1].0),
            transform_point(&world_transform, shape.positions[i2].0),
        ];
        let area = triangle_area(positions);
        if !area.is_finite() {
            return Err(PbrtError::error(&format!(
                "Area light shape node \"{name}\" contains a non-finite triangle area."
            )));
        }
        if area <= 0.0 {
            continue;
        }
        let mut geometric_normal = triangle_geometric_normal(positions)?;
        if reverse_orientation ^ transform_swaps_handedness(*world_transform) {
            geometric_normal = scale3(geometric_normal, -1.0);
        }
        total_area += area;
        bound_triangles.push(AreaTriangleInput {
            world_positions: positions,
            area,
            geometric_normal,
        });
        entries.push((primitive, area));
    }
    if entries.is_empty() || !total_area.is_finite() || total_area <= 0.0 {
        return Err(PbrtError::error(&format!(
            "Area-light shape node \"{name}\" contains no valid triangles."
        )));
    }
    let mut cumulative = 0.0;
    let mut previous_cdf = 0.0;
    for (primitive, area) in entries {
        cumulative += area / total_area;
        if cumulative <= previous_cdf {
            return Err(PbrtError::error(&format!(
                        "Area-light shape node \"{name}\" has indistinguishable adjacent CDF entries after f32 packing."
                    )));
        }
        builder
            .triangle_distributions
            .push(TriangleDistributionEntry {
                primitive,
                cdf: cumulative,
                area,
            });
        previous_cdf = cumulative;
    }
    if let Some(last) = builder.triangle_distributions.last_mut() {
        last.cdf = 1.0;
    }
    let sampling_model = u32::try_from(builder.light_sampling_models.len()).map_err(|_| {
        PbrtError::error("The flattened GPU light sampling model table exceeds u32.")
    })?;
    let alpha_zero_light = area_light_has_constant_zero_alpha(material_source, builder)?;
    builder.light_sampling_models.push(LightSamplingModel {
        kind: LightKind::Area,
        geometry_kind: LightGeometryKind::Instance,
        geometry_index: instance_index,
        direction_index: INVALID_INDEX,
        distribution_offset,
        distribution_count: u32::try_from(bound_triangles.len()).map_err(|_| {
            PbrtError::error("The flattened GPU area-light distribution exceeds u32.")
        })?,
        total_area,
        flags: u32::from(two_sided) | (u32::from(alpha_zero_light) << 1),
        world_to_light: IDENTITY_LINEAR_TRANSFORM,
    });
    let emission_attr = push_spectrum_attribute(builder, "L", &emission)?;
    let scale_attr = push_scalar_attribute(builder, "scale", scale)?;
    builder.lights.push(Light {
        kind: LightKind::Area,
        attributes: vec![emission_attr, scale_attr],
        sampling_model,
        image_index: INVALID_INDEX,
    });
    builder.light_bound_inputs.push(LightBoundInput::AreaGroup {
        handle: light_handle,
        triangles: bound_triangles,
        emission_max,
        scale,
        two_sided,
    });
    Ok(light_handle)
}

fn area_light_has_constant_zero_alpha(
    material_source: u32,
    builder: &FlatBuilder,
) -> Result<bool, PbrtError> {
    let Some(material) = builder.material_source_nodes.get(material_source as usize) else {
        return Err(PbrtError::error(
            "Area light references an invalid material source node.",
        ));
    };
    if material.kind != "alphamask" {
        return Ok(false);
    }
    let Some(alpha) = material.attributes.first() else {
        return Err(PbrtError::error(
            "Area-light AlphaMask has no alpha attribute.",
        ));
    };
    if alpha.name != "alpha" {
        return Err(PbrtError::error(
            "Area-light AlphaMask attribute is not alpha.",
        ));
    }
    match alpha.kind {
        super::AttributeKind::Scalar => {
            let value = builder
                .scalar_attributes
                .get(alpha.index as usize)
                .ok_or_else(|| PbrtError::error("Area-light alpha scalar is missing."))?;
            Ok(*value == 0.0)
        }
        super::AttributeKind::Texture => {
            let root = builder
                .texture_root_specs
                .get(alpha.index as usize)
                .ok_or_else(|| PbrtError::error("Area-light alpha texture root is missing."))?;
            let super::TextureRootSpec::Float { node } = root else {
                return Err(PbrtError::error(
                    "Area-light alpha texture root must be a float texture.",
                ));
            };
            let constant = node
                .components
                .iter()
                .find_map(|component| match component {
                    TextureComponent::Texture(texture)
                        if texture.kind == TextureKind::Float && texture.name == "constant" =>
                    {
                        Some(texture.params.get_one_float("value", 0.0) as f32)
                    }
                    _ => None,
                });
            Ok(node.children.is_empty() && constant == Some(0.0))
        }
        _ => Err(PbrtError::error(
            "Area-light alpha attribute must be scalar or float texture.",
        )),
    }
}
