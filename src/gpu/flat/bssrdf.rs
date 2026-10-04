use super::AttributeRef;
use crate::bssrdf::{compute_beam_diffusion_bssrdf, BSSRDFTable};
use crate::util::base::Float;

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

impl TabulatedBSSRDFTable {
    pub fn new(g: f32, eta: f32) -> Self {
        let mut table = BSSRDFTable::new(100, 64);
        compute_beam_diffusion_bssrdf(g as Float, eta as Float, &mut table);
        let convert = |values: Vec<Float>| values.into_iter().map(|v| v as f32).collect();
        Self {
            g,
            eta,
            rho_samples: convert(table.rho_samples),
            radius_samples: convert(table.radius_samples),
            profile: convert(table.profile),
            rho_eff: convert(table.rho_eff),
            profile_cdf: convert(table.profile_cdf),
        }
    }
}
