use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::profile::*;

#[derive(Debug, Clone)]
pub enum PhaseFunction {
    HG(HGPhaseFunction),
}

impl PhaseFunction {
    pub fn p(&self, wo: &Vector3f, wi: &Vector3f) -> Float {
        match self {
            Self::HG(phase) => phase.p(wo, wi),
        }
    }

    pub fn sample_p(&self, wo: &Vector3f, u: &Point2f) -> (Float, Vector3f) {
        match self {
            Self::HG(phase) => phase.sample_p(wo, u),
        }
    }
}

impl std::fmt::Display for PhaseFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HG(phase) => phase.fmt(f),
        }
    }
}

/// Counterpart to pbrt-v4 `HGPhaseFunction`.
#[derive(Debug, Clone, Copy)]
pub struct HGPhaseFunction {
    g: Float,
}

impl HGPhaseFunction {
    pub fn new(g: Float) -> Self {
        Self { g }
    }

    pub fn p(&self, wo: &Vector3f, wi: &Vector3f) -> Float {
        let _p = ProfilePhase::new(Prof::PhaseFuncEvaluation);
        phase_hg(Vector3f::dot(wo, wi), self.g)
    }

    pub fn sample_p(&self, wo: &Vector3f, u: &Point2f) -> (Float, Vector3f) {
        let _p = ProfilePhase::new(Prof::PhaseFuncSampling);
        // v4 `SampleHenyeyGreenstein` (sampling.cpp): clamp g into a stable
        // range and key the small-angle branch off `|g|`, otherwise negative
        // g falls into the isotropic branch and breaks back-scattering.
        let g = self.g.clamp(-0.99, 0.99);
        let cos_theta = if g.abs() < 1e-3 {
            1.0 - 2.0 * u[0]
        } else {
            let sqr_term = (1.0 - g * g) / (1.0 + g - 2.0 * g * u[0]);
            -(1.0 + g * g - sqr_term * sqr_term) / (2.0 * g)
        };

        let sin_theta = Float::max(0.0, 1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * PI * u[1];
        let (v1, v2) = coordinate_system(wo);
        let wi = spherical_direction_axes(sin_theta, cos_theta, phi, &v1, &v2, wo);
        (phase_hg(cos_theta, g), wi)
    }
}

impl From<HGPhaseFunction> for PhaseFunction {
    fn from(phase: HGPhaseFunction) -> Self {
        Self::HG(phase)
    }
}

impl std::fmt::Display for HGPhaseFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ HGPhaseFunction g: {} ]", self.g)
    }
}

// Media Inline Functions
#[inline]
pub fn phase_hg(cos_theta: Float, g: Float) -> Float {
    let denom = 1.0 + g * g + 2.0 * g * cos_theta;
    INV_4_PI * (1.0 - g * g) / (denom * Float::sqrt(denom))
}
