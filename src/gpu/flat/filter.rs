#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterKind {
    Box = 0,
    Triangle = 1,
    Gaussian = 2,
    Mitchell = 3,
    Sinc = 4,
}

/// v4 FilterSampler's signed 2D table and absolute-value sampling distribution.
/// CDFs and integrals use the unit square, as in the shared CPU FilterSampler.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterDistribution {
    pub dimensions: [u32; 2],
    pub values: Vec<f32>,
    pub conditional_cdf: Vec<f32>,
    pub marginal_cdf: Vec<f32>,
    pub integral: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Filter {
    pub kind: FilterKind,
    pub radius: [f32; 2],
    pub distribution: Option<FilterDistribution>,
}
