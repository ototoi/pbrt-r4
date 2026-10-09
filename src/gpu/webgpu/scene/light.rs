use std::collections::HashMap;

use crate::gpu::flat;
use crate::util::error::PbrtError;

use super::super::abi::{
    AttributeRef, ImageInfiniteDistributionTexel, ImageInfiniteSamplingRecord, LightRecord,
    LightSamplingModel, PortalDistributionTexel, PortalImageInfiniteRecord,
    TriangleDistributionEntry, LIGHT_KIND_AREA, LIGHT_KIND_DISTANT, LIGHT_KIND_IMAGE_INFINITE,
    LIGHT_KIND_POINT, LIGHT_KIND_PORTAL_IMAGE_INFINITE, LIGHT_KIND_SPOT,
    LIGHT_KIND_UNIFORM_INFINITE,
};

pub struct LightData {
    pub attributes: Vec<AttributeRef>,
    pub records: Vec<LightRecord>,
    pub sampling_models: Vec<LightSamplingModel>,
}

pub struct LightSamplingData {
    pub positions: Vec<[f32; 4]>,
    pub triangle_distributions: Vec<TriangleDistributionEntry>,
    pub portal_records: Vec<PortalImageInfiniteRecord>,
    pub portal_distribution: Vec<PortalDistributionTexel>,
    pub image_infinite_records: Vec<ImageInfiniteSamplingRecord>,
    pub image_infinite_distribution: Vec<ImageInfiniteDistributionTexel>,
}

impl LightSamplingData {
    pub fn from_flat(scene: &flat::Scene) -> Self {
        Self {
            positions: scene
                .light_positions
                .iter()
                .map(|position| [position[0], position[1], position[2], 1.0])
                .collect(),
            triangle_distributions: scene
                .triangle_distributions
                .iter()
                .map(|entry| TriangleDistributionEntry {
                    primitive: entry.primitive,
                    cdf: entry.cdf,
                    area: entry.area,
                    reserved: 0,
                })
                .collect(),
            portal_records: scene
                .portal_infinite_lights
                .iter()
                .map(|image| PortalImageInfiniteRecord {
                    portal: image
                        .portal
                        .map(|point| [point[0], point[1], point[2], 1.0]),
                    world_to_portal: image.world_to_portal,
                    distribution_offset: image.distribution_offset,
                    width: image.resolution[0],
                    height: image.resolution[1],
                    reserved: 0,
                })
                .collect(),
            portal_distribution: scene
                .portal_distribution
                .iter()
                .map(|value| PortalDistributionTexel {
                    function: value.function,
                    summed_area: value.summed_area,
                })
                .collect(),
            image_infinite_records: scene
                .image_infinite_lights
                .iter()
                .map(|image| ImageInfiniteSamplingRecord {
                    distribution_offset: image.distribution_offset,
                    row_cdf_offset: image.row_cdf_offset,
                    width: image.resolution[0],
                    height: image.resolution[1],
                    light_to_render: image.light_to_render,
                })
                .collect(),
            image_infinite_distribution: scene
                .image_infinite_distribution
                .iter()
                .map(|texel| ImageInfiniteDistributionTexel {
                    weight: texel.weight,
                    conditional_cdf: texel.conditional_cdf,
                })
                .collect(),
        }
    }
}

pub fn validate_instance_area_lights(
    instance_index: usize,
    instance: &flat::Instance,
    flat: &flat::Scene,
) -> Result<(), PbrtError> {
    if instance.area_light == flat::INVALID_INDEX {
        return Ok(());
    }
    let geometry = flat
        .geometries
        .get(instance.geometry as usize)
        .ok_or_else(|| PbrtError::error("Flat area-light instance has an invalid geometry."))?;
    let triangle_count = geometry.index_count / 3;
    if triangle_count == 0 {
        return Err(PbrtError::error(
            "Flat area-light instance geometry contains no triangles.",
        ));
    }
    let handle = instance.area_light;
    let record = flat.lights.get(handle as usize).ok_or_else(|| {
        PbrtError::error(&format!(
            "Flat instance {instance_index} has an invalid area-light handle."
        ))
    })?;
    if record.kind != flat::LightKind::Area {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light range contains a non-area light."
        )));
    }
    let model = flat
        .light_sampling_models
        .get(record.sampling_model as usize)
        .ok_or_else(|| {
            PbrtError::error(&format!(
                "Flat instance {instance_index} references an invalid area-light payload."
            ))
        })?;
    if model.geometry_index as usize != instance_index {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light range does not match its triangles."
        )));
    }
    let offset = usize::try_from(model.distribution_offset)
        .map_err(|_| PbrtError::error("Flat area-light distribution offset does not fit usize."))?;
    let count = usize::try_from(model.distribution_count)
        .map_err(|_| PbrtError::error("Flat area-light distribution count does not fit usize."))?;
    if count == 0 {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light distribution is empty."
        )));
    }
    let end = offset
        .checked_add(count)
        .ok_or_else(|| PbrtError::error("Flat area-light distribution range overflowed."))?;
    let entries = flat
        .triangle_distributions
        .get(offset..end)
        .ok_or_else(|| {
            PbrtError::error(&format!(
                "Flat instance {instance_index} area-light distribution range is invalid."
            ))
        })?;
    if !model.total_area.is_finite() || model.total_area <= 0.0 {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light total area is invalid."
        )));
    }
    let mut previous_cdf = 0.0;
    let mut area_sum = 0.0;
    for entry in entries {
        if entry.primitive >= triangle_count
            || !entry.area.is_finite()
            || entry.area <= 0.0
            || !entry.cdf.is_finite()
            || entry.cdf < previous_cdf
            || entry.cdf > 1.0
        {
            return Err(PbrtError::error(&format!(
                "Flat instance {instance_index} has an invalid area-light distribution entry."
            )));
        }
        previous_cdf = entry.cdf;
        area_sum += entry.area;
    }
    if previous_cdf != 1.0 || (area_sum - model.total_area).abs() > model.total_area.abs() * 1e-5 {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light distribution does not match total area."
        )));
    }
    Ok(())
}

pub fn convert_lights(
    light_sampling_models: &[flat::LightSamplingModel],
    lights: &[flat::Light],
    infinite_lights: &[flat::Light],
    mut attributes: Vec<AttributeRef>,
    infinite_image_payloads: &HashMap<u32, u32>,
) -> Result<LightData, PbrtError> {
    let all_lights = lights
        .iter()
        .chain(infinite_lights.iter())
        .collect::<Vec<_>>();
    let mut light_attribute_offsets = Vec::with_capacity(all_lights.len());
    let mut next_attribute_offset = u32::try_from(attributes.len())
        .map_err(|_| PbrtError::error("Flat light attribute offset exceeds u32."))?;
    for light in &all_lights {
        light_attribute_offsets.push(next_attribute_offset);
        let attribute_count = u32::try_from(light.attributes.len())
            .map_err(|_| PbrtError::error("Flat light attribute count exceeds u32."))?;
        next_attribute_offset = next_attribute_offset
            .checked_add(attribute_count)
            .ok_or_else(|| PbrtError::error("Flat light attribute offsets overflow u32."))?;
        attributes.extend(light.attributes.iter().map(|attribute| AttributeRef {
            kind: match attribute.kind {
                flat::AttributeKind::Scalar => 0,
                flat::AttributeKind::Spectrum => 1,
                flat::AttributeKind::Texture => 2,
                flat::AttributeKind::Measured => 3,
            },
            index: attribute.index,
        }));
    }

    let sampling_models = light_sampling_models
        .iter()
        .enumerate()
        .map(|(model_index, model)| {
            let model_index = u32::try_from(model_index)
                .map_err(|_| PbrtError::error("Flat light sampling model index exceeds u32."))?;
            Ok(LightSamplingModel {
                kind: light_kind(model.kind),
                geometry_kind: match model.geometry_kind {
                    flat::LightGeometryKind::Position => 0,
                    flat::LightGeometryKind::Instance => 1,
                    flat::LightGeometryKind::Direction => 2,
                    flat::LightGeometryKind::Portal => 3,
                    flat::LightGeometryKind::ImageInfinite => 4,
                },
                geometry_index: model.geometry_index,
                direction_index: model.direction_index,
                distribution_offset_words: model.distribution_offset,
                distribution_count: model.distribution_count,
                total_area: model.total_area,
                flags: infinite_image_payloads
                    .get(&model_index)
                    .copied()
                    .unwrap_or(model.flags),
                world_to_light: model.world_to_light,
            })
        })
        .collect::<Result<Vec<_>, PbrtError>>()?;

    let records = lights
        .iter()
        .enumerate()
        .chain(
            infinite_lights
                .iter()
                .enumerate()
                .map(|(index, light)| (lights.len() + index, light)),
        )
        .map(|(light_index, light)| {
            let attribute_count = u32::try_from(light.attributes.len())
                .map_err(|_| PbrtError::error("Flat light attribute count exceeds u32."))?;
            Ok(LightRecord {
                kind: light_kind(light.kind),
                attribute_offset: light_attribute_offsets[light_index],
                attribute_count,
                sampling_model: light.sampling_model,
            })
        })
        .collect::<Result<Vec<_>, PbrtError>>()?;

    Ok(LightData {
        attributes,
        records,
        sampling_models,
    })
}

fn light_kind(kind: flat::LightKind) -> u32 {
    match kind {
        flat::LightKind::Point => LIGHT_KIND_POINT,
        flat::LightKind::Spot => LIGHT_KIND_SPOT,
        flat::LightKind::Area => LIGHT_KIND_AREA,
        flat::LightKind::Distant => LIGHT_KIND_DISTANT,
        flat::LightKind::UniformInfinite => LIGHT_KIND_UNIFORM_INFINITE,
        flat::LightKind::ImageInfinite => LIGHT_KIND_IMAGE_INFINITE,
        flat::LightKind::PortalImageInfinite => LIGHT_KIND_PORTAL_IMAGE_INFINITE,
    }
}
