use super::portal::{PortalDistributionTexel, PortalImageInfiniteLight};
use super::texture::TextureLibrary;
use super::{
    Camera, DenseSpectrum, Film, Geometry, Instance, Light, LightBVH, LightBounds,
    LightGeometryKind, LightKind, LightSamplingModel, MaterialNode, MaterialRoot,
    MeasuredBsdfResources, Medium, Output, PrimitiveDistributionMap, RenderSettings,
    TriangleDistributionEntry, Vertex, Viewport, INVALID_INDEX,
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
    pub scalar_attributes: Vec<f32>,
    /// Typed, backend-independent texture programs and shared image resources.
    pub texture_library: TextureLibrary,
    pub spectrum_attributes: Vec<DenseSpectrum>,
    pub measured_bsdfs: MeasuredBsdfResources,
    pub primitive_distribution_map: PrimitiveDistributionMap,
    pub portal_infinite_lights: Vec<PortalImageInfiniteLight>,
    pub portal_distribution: Vec<PortalDistributionTexel>,
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
        for model in &self.light_sampling_models {
            if (model.kind == LightKind::PortalImageInfinite)
                != (model.geometry_kind == LightGeometryKind::Portal)
            {
                return Err(PbrtError::error(
                    "Portal sampling model kind and geometry kind disagree.",
                ));
            }
        }
        Ok(())
    }
}
