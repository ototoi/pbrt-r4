#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettings {
    pub samples_per_pixel: u32,
    pub max_depth: u32,
    pub seed: u32,
}
