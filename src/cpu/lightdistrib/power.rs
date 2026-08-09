use super::lightdistrib::*;
use crate::cpu::integrators::IntegratorBase;
use crate::util::base::*;
use crate::util::sampling::*;
use crate::util::spectrum::{safe_div, SampledWavelengths};

use std::sync::Arc;

/// pbrt-v4 `PowerLightSampler` constructor body (lightsamplers.cpp:76-93)
/// computes each light's power as
/// `SafeDiv(light.Phi(lambda), lambda.PDF()).Average()`. The
/// `lambda.PDF()` division converts the importance-sampled Phi back to
/// a uniform spectral density so that lights with different spectral
/// distributions (e.g. blackbody at different temperatures) are
/// compared on the same footing. Skipping the division biases the
/// sampler pmf toward whichever spectral region the wavelength sampler
/// happened to weight; this manifests as a per-light beta error in
/// BDPT / Path tracers (works/20260527/bdpt-strategy-isolation/).
pub fn compute_light_power_distribution(base: &IntegratorBase) -> Arc<Distribution1D> {
    assert!(!base.lights.is_empty());
    let mut light_power = Vec::new();
    let lambda = SampledWavelengths::sample_visible(0.5);
    let lambda_pdf = lambda.pdf_spectrum();
    for light in base.lights.iter() {
        let light = light.as_ref();
        let phi = light.phi(&lambda);
        let normalized = safe_div(phi, lambda_pdf);
        light_power.push(Float::max(0.0, normalized.average()));
    }
    return Arc::new(Distribution1D::new(&light_power));
}

pub struct PowerLightDistribution {
    distrib: Arc<Distribution1D>,
}

impl PowerLightDistribution {
    pub fn new(base: &IntegratorBase) -> Self {
        PowerLightDistribution {
            distrib: compute_light_power_distribution(base),
        }
    }
}

impl LightDistribution for PowerLightDistribution {
    fn lookup(&self, _p: &Point3f) -> Arc<Distribution1D> {
        return self.distrib.clone();
    }
}
