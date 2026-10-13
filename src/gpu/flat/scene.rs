use super::texture::TextureLibrary;
use super::{
    AttributeResources, Camera, Film, GeometryResources, Instance, LightResources,
    MaterialResources, Medium, Output, RenderSettings, Viewport, INVALID_INDEX,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub camera: Camera,
    pub viewport: Viewport,
    pub film: Film,
    pub output: Output,
    pub render_settings: RenderSettings,
    pub lights: LightResources,
    /// Packed geometry arrays addressed by `Instance::geometry`.
    pub geometry: GeometryResources,
    pub instances: Vec<Instance>,
    pub media: Vec<Medium>,
    pub materials: MaterialResources,
    /// Scene-wide scalar and spectrum tables referenced by materials, lights, and media.
    pub attributes: AttributeResources,
    /// Typed, backend-independent texture programs and shared image resources.
    pub texture_library: TextureLibrary,
}
