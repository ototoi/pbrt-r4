pub const MAX_GPU_RENDER_DEPTH: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerKind {
    Independent,
    Halton,
    Sobol,
    PaddedSobol,
    ZSobol,
    Pmj02Bn,
    Stratified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerRandomization {
    None,
    PermuteDigits,
    FastOwen,
    Owen,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettings {
    pub sampler_kind: SamplerKind,
    pub randomization: SamplerRandomization,
    pub samples_per_pixel: u32,
    pub x_samples: u32,
    pub y_samples: u32,
    pub jitter: bool,
    pub max_depth: u32,
    pub seed: u32,
    pub light_sampler: String,
    pub disable_wavelength_jitter: bool,
}
