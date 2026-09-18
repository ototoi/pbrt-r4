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
    pub infinite_lights: Vec<Light>,
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
    pub texture_roots: Vec<TextureRootRecord>,
    pub image_views: Vec<crate::gpu::texture::ImageView>,
    pub texture_nodes: Vec<TextureNode>,
    pub texture_child_indices: Vec<u32>,
    pub spectrum_attributes: Vec<DenseSpectrum>,
    pub primitive_distribution_map: PrimitiveDistributionMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Lowered material entry point into a texture graph.
///
/// Unlike the Node IR specification, this record contains only table indices
/// and is therefore directly consumable by backend adapters.
pub struct TextureRootRecord {
    pub texture_node: u32,
    /// 0=Albedo, 1=Unbounded, 2=Illuminant. Ignored for Float roots.
    pub spectrum_type: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextureNode {
    pub name: String,
    pub kind: u32,
    pub implementation: String,
    pub first_child: u32,
    pub child_count: u32,
    pub image_view: Option<u32>,
    pub mapping: [f32; 16],
    /// RGB colour space used when converting a spectrum texture at the
    /// material boundary (0=sRGB, 1=ACES2065-1, 2=DCI-P3, 3=Rec.2020).
    pub color_space: u32,
    pub operation: u32,
    /// Mapping kind: 0=UV, 1=planar, 2=spherical, 3=cylindrical.
    pub mapping_kind: u32,
    pub constant_value: [f32; 4],
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
