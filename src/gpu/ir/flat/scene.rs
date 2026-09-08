use super::{
    AttributeRef, Camera, DenseSpectrum, Film, Geometry, Instance, Light, LightBVH, LightBounds,
    LightKind, LightSamplingModel, Material, Output, PrimitiveDistributionMap, RenderSettings,
    TriangleDistributionEntry, Vertex, Viewport,
};

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
    /// One upload arena shared by material and light attribute references.
    pub attribute_refs: Vec<AttributeRef>,
    pub light_bounds: Vec<LightBounds>,
    pub light_bvh: LightBVH,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<Instance>,
    pub materials: Vec<Material>,
    pub scalar_attributes: Vec<f32>,
    pub texture_attributes: Vec<u32>,
    pub spectrum_attributes: Vec<DenseSpectrum>,
    pub primitive_distribution_map: PrimitiveDistributionMap,
}

impl Scene {
    pub fn validate_static_views(&self) -> Result<(), crate::util::error::PbrtError> {
        let area_count = self
            .lights
            .iter()
            .filter(|light| light.kind == LightKind::Area)
            .count();
        if self.primitive_distribution_map.offsets.len() != area_count + 1 {
            return Err(crate::util::error::PbrtError::error(
                "Primitive distribution map offsets do not match area lights.",
            ));
        }
        let last = *self.primitive_distribution_map.offsets.last().unwrap_or(&0) as usize;
        if last != self.primitive_distribution_map.entries.len() {
            return Err(crate::util::error::PbrtError::error(
                "Primitive distribution map range is inconsistent.",
            ));
        }
        Ok(())
    }
}
