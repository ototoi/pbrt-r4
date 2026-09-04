use super::{
    AreaLight, Camera, Geometry, Instance, LightRecord, Material, Output, PointLight,
    RenderSettings, Vertex, Viewport,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub camera: Camera,
    pub viewport: Viewport,
    pub output: Output,
    pub render_settings: RenderSettings,
    pub point_lights: Vec<PointLight>,
    pub area_lights: Vec<AreaLight>,
    pub lights: Vec<LightRecord>,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<Instance>,
    pub materials: Vec<Material>,
}
