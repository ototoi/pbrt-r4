use crate::gpu::ir::flat;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderSettings {
    pub samples_per_pixel: u32,
    pub max_depth: u32,
}

impl RenderSettings {
    pub fn from_flat(settings: flat::RenderSettings) -> Self {
        Self {
            samples_per_pixel: settings.samples_per_pixel,
            max_depth: settings.max_depth,
        }
    }
}
