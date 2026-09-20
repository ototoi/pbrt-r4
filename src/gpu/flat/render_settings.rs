pub const MAX_GPU_RENDER_DEPTH: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerKind {
    Independent,
    Halton,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HaltonRandomization {
    None,
    PermuteDigits,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettings {
    pub sampler_kind: SamplerKind,
    pub halton_randomization: HaltonRandomization,
    pub samples_per_pixel: u32,
    pub max_depth: u32,
    pub seed: u32,
    pub light_sampler: String,
    pub disable_wavelength_jitter: bool,
}
