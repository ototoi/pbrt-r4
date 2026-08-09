use crate::base::bxdf::*;
use crate::util::base::*;
use crate::util::sampling::{cosine_hemisphere_pdf, cosine_sample_hemisphere};
use crate::util::scattering::{abs_cos_theta, cos_theta, same_hemisphere};
use crate::util::spectrum::*;

/// Direct translation of pbrt-v4 `NormalizedFresnelBxDF`
/// (`bxdfs.h:1073`). All evaluation produces `SampledSpectrum` directly
/// — there is no dense `Spectrum` reflectance to store.
#[derive(Debug, Clone, Copy)]
pub struct NormalizedFresnelBxDF {
    eta: Float,
}

impl NormalizedFresnelBxDF {
    pub fn new(eta: Float) -> Self {
        Self { eta }
    }

    /// pbrt-v4 `NormalizedFresnelBxDF::f`.
    pub fn f(&self, wo: &Vector3f, wi: &Vector3f, mode: TransportMode) -> SampledSpectrum {
        if !same_hemisphere(wo, wi) {
            return SampledSpectrum::zero();
        }
        // Compute $\Sw$ factor for BSSRDF value.
        let c = 1.0 - 2.0 * fresnel_moment1(1.0 / self.eta);
        let mut f = SampledSpectrum::new((1.0 - fr_dielectric(cos_theta(wi), self.eta)) / (c * PI));
        // Update BSSRDF transmission term to account for adjoint light transport.
        if mode == TransportMode::Radiance {
            f *= self.eta * self.eta;
        }
        f
    }

    /// pbrt-v4 `NormalizedFresnelBxDF::Sample_f`.
    pub fn sample_f(
        &self,
        wo: &Vector3f,
        _uc: Float,
        u: &Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            return None;
        }

        let mut wi = cosine_sample_hemisphere(u);
        if wo.z < 0.0 {
            wi.z *= -1.0;
        }
        Some(BSDFSample::new(
            self.f(wo, &wi, mode),
            wi,
            self.pdf(wo, &wi, mode, sample_flags),
            BXDF_REFLECTION | BXDF_DIFFUSE,
            1.0,
            false,
        ))
    }

    /// pbrt-v4 `NormalizedFresnelBxDF::PDF`.
    pub fn pdf(
        &self,
        wo: &Vector3f,
        wi: &Vector3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            return 0.0;
        }
        if same_hemisphere(wo, wi) {
            abs_cos_theta(wi) * INV_PI
        } else {
            0.0
        }
    }

    /// pbrt-v4 `NormalizedFresnelBxDF::Flags`.
    pub fn flags(&self) -> BxDFFlags {
        BXDF_REFLECTION | BXDF_DIFFUSE
    }

    /// pbrt-v4 `NormalizedFresnelBxDF::Regularize` — no-op.
    pub fn regularize(&mut self) {}
}

/// pbrt-v4 `FresnelMoment1` from `util/scattering.cpp`.
fn fresnel_moment1(eta: Float) -> Float {
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let eta4 = eta3 * eta;
    let eta5 = eta4 * eta;
    if eta < 1.0 {
        0.45966 - 1.73965 * eta + 3.37668 * eta2 - 3.904945 * eta3 + 2.49277 * eta4 - 0.68441 * eta5
    } else {
        -4.61686 + 11.1136 * eta - 10.4646 * eta2 + 5.11455 * eta3 - 1.27198 * eta4 + 0.12746 * eta5
    }
}

/// pbrt-v4 `FrDielectric(cosTheta_i, eta)` from `util/scattering.h`. The
/// 2-argument form folds the entering / exiting branches; `eta` already
/// encodes the relative index.
fn fr_dielectric(mut cos_theta_i: Float, mut eta: Float) -> Float {
    cos_theta_i = Float::clamp(cos_theta_i, -1.0, 1.0);
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
    }

    let sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i;
    let sin2_theta_t = sin2_theta_i / (eta * eta);
    if sin2_theta_t >= 1.0 {
        return 1.0;
    }
    let cos_theta_t = Float::sqrt(Float::max(0.0, 1.0 - sin2_theta_t));
    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
    (r_parl * r_parl + r_perp * r_perp) * 0.5
}
