use super::AttributeRef;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BSSRDFCoefficientKind {
    Sigma,
    ReflectanceMfp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BSSRDF {
    pub scale: f32,
    pub g: f32,
    pub eta: f32,
    pub table_index: u32,
    pub coefficient_kind: BSSRDFCoefficientKind,
    pub coefficients: [AttributeRef; 2],
}

/// pbrt-v4 `BSSRDFTable`, generated on the host and stored at GPU precision.
#[derive(Clone, Debug, PartialEq)]
pub struct TabulatedBSSRDFTable {
    pub g: f32,
    pub eta: f32,
    pub rho_samples: Vec<f32>,
    pub radius_samples: Vec<f32>,
    pub profile: Vec<f32>,
    pub rho_eff: Vec<f32>,
    pub profile_cdf: Vec<f32>,
}
