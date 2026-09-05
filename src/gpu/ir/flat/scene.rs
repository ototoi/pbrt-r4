use super::{
    AreaLight, Camera, Geometry, Instance, LightBVH, LightBounds, LightRecord, Material, Output,
    PointLight, RenderSettings, TriangleDistributionEntry, Vertex, Viewport,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub camera: Camera,
    pub viewport: Viewport,
    pub output: Output,
    pub render_settings: RenderSettings,
    pub point_lights: Vec<PointLight>,
    pub area_lights: Vec<AreaLight>,
    pub triangle_distributions: Vec<TriangleDistributionEntry>,
    pub lights: Vec<LightRecord>,
    pub light_bounds: Vec<LightBounds>,
    pub light_bvh: LightBVH,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<Instance>,
    pub materials: Vec<Material>,
}
