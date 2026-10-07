use super::portal::{PortalDistributionTexel, PortalImageInfiniteLight};
use super::texture::TextureLibrary;
use super::{
    Camera, DenseSpectrum, Film, Geometry, ImageInfiniteDistributionTexel,
    ImageInfiniteSamplingRecord, Instance, Light, LightBVH, LightBounds, LightGeometryKind,
    LightKind, LightSamplingModel, MaterialNode, MaterialRoot, MeasuredBsdfResources, Medium,
    Output, PrimitiveDistributionMap, RenderSettings, TabulatedBSSRDFTable,
    TriangleDistributionEntry, Vertex, Viewport, BSSRDF, INVALID_INDEX,
};
use crate::util::error::PbrtError;

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub camera: Camera,
    pub viewport: Viewport,
    pub film: Film,
    pub output: Output,
    pub render_settings: RenderSettings,
    pub light_sampling_models: Vec<LightSamplingModel>,
    pub light_positions: Vec<[f32; 3]>,
    pub triangle_distributions: Vec<TriangleDistributionEntry>,
    pub lights: Vec<Light>,
    pub infinite_lights: Vec<Light>,
    pub light_bounds: Vec<LightBounds>,
    pub light_bvh: LightBVH,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<Instance>,
    pub media: Vec<Medium>,
    pub material_roots: Vec<MaterialRoot>,
    pub material_nodes: Vec<MaterialNode>,
    pub bssrdfs: Vec<BSSRDF>,
    pub bssrdf_tables: Vec<TabulatedBSSRDFTable>,
    pub scalar_attributes: Vec<f32>,
    /// Typed, backend-independent texture programs and shared image resources.
    pub texture_library: TextureLibrary,
    pub spectrum_attributes: Vec<DenseSpectrum>,
    pub measured_bsdfs: MeasuredBsdfResources,
    pub primitive_distribution_map: PrimitiveDistributionMap,
    pub portal_infinite_lights: Vec<PortalImageInfiniteLight>,
    pub portal_distribution: Vec<PortalDistributionTexel>,
    pub image_infinite_lights: Vec<ImageInfiniteSamplingRecord>,
    pub image_infinite_distribution: Vec<ImageInfiniteDistributionTexel>,
    pub image_infinite_row_cdf: Vec<f32>,
}

impl Scene {
    pub fn validate_static_views(&self) -> Result<(), PbrtError> {
        let area_count = self
            .lights
            .iter()
            .filter(|light| light.kind == LightKind::Area)
            .count();
        if self.primitive_distribution_map.offsets.len() != area_count + 1 {
            return Err(PbrtError::error(
                "Primitive distribution map offsets do not match area lights.",
            ));
        }
        let last = *self.primitive_distribution_map.offsets.last().unwrap_or(&0) as usize;
        if last != self.primitive_distribution_map.entries.len() {
            return Err(PbrtError::error(
                "Primitive distribution map range is inconsistent.",
            ));
        }
        let portal_light_count = self
            .infinite_lights
            .iter()
            .filter(|light| light.kind == LightKind::PortalImageInfinite)
            .count();
        if portal_light_count != self.portal_infinite_lights.len() {
            return Err(PbrtError::error(
                "Portal infinite lights and geometry records are not one-to-one.",
            ));
        }

        let mut next_distribution_offset = 0usize;
        for (index, portal) in self.portal_infinite_lights.iter().enumerate() {
            if portal.resolution.contains(&0) {
                return Err(PbrtError::error(&format!(
                    "Portal infinite light {index} has an empty distribution."
                )));
            }
            if portal
                .portal
                .iter()
                .flatten()
                .chain(portal.world_to_portal.iter().flatten())
                .any(|value| !value.is_finite())
            {
                return Err(PbrtError::error(&format!(
                    "Portal infinite light {index} contains non-finite geometry."
                )));
            }
            let count = portal.resolution[0]
                .checked_mul(portal.resolution[1])
                .and_then(|count| usize::try_from(count).ok())
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Portal infinite light {index} distribution size overflowed."
                    ))
                })?;
            let offset = usize::try_from(portal.distribution_offset).map_err(|_| {
                PbrtError::error(&format!(
                    "Portal infinite light {index} distribution offset is invalid."
                ))
            })?;
            if offset != next_distribution_offset
                || offset > self.portal_distribution.len()
                || count > self.portal_distribution.len() - offset
            {
                return Err(PbrtError::error(&format!(
                    "Portal infinite light {index} distribution range is invalid."
                )));
            }
            let end = offset + count;
            if self.portal_distribution[offset..end]
                .iter()
                .any(|texel| !texel.function.is_finite() || !texel.summed_area.is_finite())
            {
                return Err(PbrtError::error(&format!(
                    "Portal infinite light {index} distribution contains a non-finite value."
                )));
            }
            next_distribution_offset = end;
        }
        if next_distribution_offset != self.portal_distribution.len() {
            return Err(PbrtError::error(
                "Portal distribution contains texels not owned by a portal light.",
            ));
        }

        let image_light_count = self
            .infinite_lights
            .iter()
            .filter(|light| light.kind == LightKind::ImageInfinite)
            .count();
        if image_light_count != self.image_infinite_lights.len() {
            return Err(PbrtError::error(
                "Image infinite lights and sampling records are not one-to-one.",
            ));
        }
        let mut next_image_texel_offset = 0usize;
        let mut next_image_row_offset = 0usize;
        for (index, image) in self.image_infinite_lights.iter().enumerate() {
            let [width, height] = image.resolution;
            if width == 0 || height == 0 {
                return Err(PbrtError::error(&format!(
                    "Image infinite light {index} has an empty sampling distribution."
                )));
            }
            if image
                .light_to_render
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
            {
                return Err(PbrtError::error(&format!(
                    "Image infinite light {index} has a non-finite transform."
                )));
            }
            let texel_count = width
                .checked_mul(height)
                .and_then(|count| usize::try_from(count).ok())
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Image infinite light {index} distribution size overflowed."
                    ))
                })?;
            let texel_offset = usize::try_from(image.distribution_offset).map_err(|_| {
                PbrtError::error(&format!(
                    "Image infinite light {index} distribution offset is invalid."
                ))
            })?;
            let row_offset = usize::try_from(image.row_cdf_offset).map_err(|_| {
                PbrtError::error(&format!(
                    "Image infinite light {index} row CDF offset is invalid."
                ))
            })?;
            let texel_end = texel_offset.checked_add(texel_count).ok_or_else(|| {
                PbrtError::error("Image infinite light distribution range overflowed.")
            })?;
            let row_end = row_offset.checked_add(height as usize).ok_or_else(|| {
                PbrtError::error("Image infinite light row CDF range overflowed.")
            })?;
            if texel_offset != next_image_texel_offset
                || row_offset != next_image_row_offset
                || texel_end > self.image_infinite_distribution.len()
                || row_end > self.image_infinite_row_cdf.len()
            {
                return Err(PbrtError::error(&format!(
                    "Image infinite light {index} distribution range is invalid."
                )));
            }
            let texels = &self.image_infinite_distribution[texel_offset..texel_end];
            for row in texels.chunks_exact(width as usize) {
                let mut previous = 0.0f32;
                for texel in row {
                    if !texel.weight.is_finite()
                        || texel.weight < 0.0
                        || !texel.conditional_cdf.is_finite()
                        || texel.conditional_cdf < previous
                    {
                        return Err(PbrtError::error(&format!(
                            "Image infinite light {index} has an invalid conditional CDF."
                        )));
                    }
                    previous = texel.conditional_cdf;
                }
            }
            let mut previous = 0.0f32;
            for value in &self.image_infinite_row_cdf[row_offset..row_end] {
                if !value.is_finite() || *value < previous {
                    return Err(PbrtError::error(&format!(
                        "Image infinite light {index} has an invalid marginal CDF."
                    )));
                }
                previous = *value;
            }
            if previous <= 0.0 {
                return Err(PbrtError::error(&format!(
                    "Image infinite light {index} has a zero sampling integral."
                )));
            }
            next_image_texel_offset = texel_end;
            next_image_row_offset = row_end;
        }
        if next_image_texel_offset != self.image_infinite_distribution.len()
            || next_image_row_offset != self.image_infinite_row_cdf.len()
        {
            return Err(PbrtError::error(
                "Image infinite distribution data is not owned by a sampling record.",
            ));
        }

        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let mut referenced_portals = vec![false; self.portal_infinite_lights.len()];
        for (light_index, light) in self.infinite_lights.iter().enumerate() {
            if light.kind != LightKind::PortalImageInfinite {
                continue;
            }
            let model = self
                .light_sampling_models
                .get(light.sampling_model as usize)
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Portal infinite light {light_index} has an invalid sampling model."
                    ))
                })?;
            if model.kind != LightKind::PortalImageInfinite
                || model.geometry_kind != LightGeometryKind::Portal
            {
                return Err(PbrtError::error(
                    "Portal infinite light has a non-Portal sampling model.",
                ));
            }
            if model.direction_index != INVALID_INDEX
                || model.distribution_offset != 0
                || model.distribution_count != 0
                || model.total_area != 0.0
                || model.world_to_light != identity
            {
                return Err(PbrtError::error(
                    "Portal light sampling model has invalid legacy geometry fields.",
                ));
            }
            let portal_index = model.geometry_index as usize;
            let portal = self
                .portal_infinite_lights
                .get(portal_index)
                .ok_or_else(|| {
                    PbrtError::error("Portal light sampling model has an invalid geometry index.")
                })?;
            if referenced_portals[portal_index] {
                return Err(PbrtError::error(
                    "Multiple Portal lights reference the same geometry record.",
                ));
            }
            referenced_portals[portal_index] = true;

            let level = self
                .texture_library
                .mipmaps
                .get(light.image_index as usize)
                .and_then(|mipmap| mipmap.levels.first())
                .ok_or_else(|| {
                    PbrtError::error("Portal infinite light references an invalid image.")
                })?;
            if level.resolution != portal.resolution {
                return Err(PbrtError::error(
                    "Portal image and sampling distribution resolutions differ.",
                ));
            }
        }
        if referenced_portals.iter().any(|referenced| !referenced) {
            return Err(PbrtError::error(
                "Portal geometry record is not referenced by exactly one light.",
            ));
        }
        let mut referenced_image_lights = vec![false; self.image_infinite_lights.len()];
        for (light_index, light) in self.infinite_lights.iter().enumerate() {
            if light.kind != LightKind::ImageInfinite {
                continue;
            }
            let model = self
                .light_sampling_models
                .get(light.sampling_model as usize)
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Image infinite light {light_index} has an invalid sampling model."
                    ))
                })?;
            if model.kind != LightKind::ImageInfinite
                || model.geometry_kind != LightGeometryKind::ImageInfinite
            {
                return Err(PbrtError::error(
                    "Image infinite light has a non-image sampling model.",
                ));
            }
            let image_index = model.geometry_index as usize;
            let image = self.image_infinite_lights.get(image_index).ok_or_else(|| {
                PbrtError::error("Image infinite sampling model has an invalid record index.")
            })?;
            if referenced_image_lights[image_index] {
                return Err(PbrtError::error(
                    "Multiple image infinite lights reference the same sampling record.",
                ));
            }
            referenced_image_lights[image_index] = true;
            let level = self
                .texture_library
                .mipmaps
                .get(light.image_index as usize)
                .and_then(|mipmap| mipmap.levels.first())
                .ok_or_else(|| {
                    PbrtError::error("Image infinite light references an invalid image.")
                })?;
            if level.resolution != image.resolution {
                return Err(PbrtError::error(
                    "Image infinite light and sampling distribution resolutions differ.",
                ));
            }
        }
        if referenced_image_lights.iter().any(|referenced| !referenced) {
            return Err(PbrtError::error(
                "Image infinite sampling record is not referenced by exactly one light.",
            ));
        }
        for model in &self.light_sampling_models {
            if (model.kind == LightKind::PortalImageInfinite)
                != (model.geometry_kind == LightGeometryKind::Portal)
            {
                return Err(PbrtError::error(
                    "Portal sampling model kind and geometry kind disagree.",
                ));
            }
            if (model.kind == LightKind::ImageInfinite)
                != (model.geometry_kind == LightGeometryKind::ImageInfinite)
            {
                return Err(PbrtError::error(
                    "Image infinite sampling model kind and geometry kind disagree.",
                ));
            }
        }
        Ok(())
    }
}
