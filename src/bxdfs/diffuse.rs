use crate::base::bxdf::*;
use crate::util::base::*;
use crate::util::sampling::*;
use crate::util::scattering::{abs_cos_theta, same_hemisphere};
use crate::util::spectrum::*;

/// Direct translation of pbrt-v4 `DiffuseBxDF` (`bxdfs.h:33`).
/// `R` is stored as a `SampledSpectrum` (the reflectance pre-sampled
/// at the path's wavelengths); v4 has no dense `Spectrum` here, only
/// the per-λ packet.
#[derive(Debug, Clone, Copy)]
pub struct DiffuseBxDF {
    r: SampledSpectrum,
}

impl DiffuseBxDF {
    pub fn new(r: SampledSpectrum) -> Self {
        DiffuseBxDF { r }
    }

    /// pbrt-v4 `DiffuseBxDF::f`.
    pub fn f(&self, wo: &Vector3f, wi: &Vector3f, _mode: TransportMode) -> SampledSpectrum {
        if !same_hemisphere(wo, wi) {
            return SampledSpectrum::zero();
        }
        self.r * INV_PI
    }

    /// pbrt-v4 `DiffuseBxDF::Sample_f`.
    pub fn sample_f(
        &self,
        wo: &Vector3f,
        _uc: Float,
        u: &Point2f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            return None;
        }
        let mut wi = cosine_sample_hemisphere(u);
        if wo.z < 0.0 {
            wi.z *= -1.0;
        }
        let pdf = cosine_hemisphere_pdf(abs_cos_theta(&wi));
        Some(BSDFSample::new(
            self.r * INV_PI,
            wi,
            pdf,
            BXDF_REFLECTION | BXDF_DIFFUSE,
            1.0,
            false,
        ))
    }

    /// pbrt-v4 `DiffuseBxDF::PDF`.
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
        if !same_hemisphere(wo, wi) {
            return 0.0;
        }
        cosine_hemisphere_pdf(abs_cos_theta(wi))
    }

    /// pbrt-v4 `DiffuseBxDF::Flags`.
    pub fn flags(&self) -> BxDFFlags {
        if self.r.is_black() {
            BXDF_UNSET
        } else {
            BXDF_REFLECTION | BXDF_DIFFUSE
        }
    }

    /// pbrt-v4 `DiffuseBxDF::Regularize` — no-op.
    pub fn regularize(&mut self) {}
}

/// Direct translation of pbrt-v4 `DiffuseTransmissionBxDF`
/// (`bxdfs.h:85`). Holds reflectance / transmittance pre-sampled at
/// the path's wavelengths; v4 has no dense `Spectrum` here.
#[derive(Debug, Clone, Copy)]
pub struct DiffuseTransmissionBxDF {
    pub r: SampledSpectrum,
    pub t: SampledSpectrum,
}

impl DiffuseTransmissionBxDF {
    pub fn new(r: SampledSpectrum, t: SampledSpectrum) -> Self {
        DiffuseTransmissionBxDF { r, t }
    }

    /// pbrt-v4 `DiffuseTransmissionBxDF::f`.
    pub fn f(&self, wo: &Vector3f, wi: &Vector3f, _mode: TransportMode) -> SampledSpectrum {
        if same_hemisphere(wo, wi) {
            self.r * INV_PI
        } else {
            self.t * INV_PI
        }
    }

    /// pbrt-v4 `DiffuseTransmissionBxDF::Sample_f`.
    pub fn sample_f(
        &self,
        wo: &Vector3f,
        uc: Float,
        u: &Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        let mut pr = self.r.max_component_value();
        let mut pt = self.t.max_component_value();
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            pr = 0.0;
        }
        if sample_flags & BXDF_REFL_TRANS_TRANSMISSION == 0 {
            pt = 0.0;
        }
        if pr == 0.0 && pt == 0.0 {
            return None;
        }

        if uc < pr / (pr + pt) {
            // Sample diffuse BSDF reflection.
            let mut wi = cosine_sample_hemisphere(u);
            if wo.z < 0.0 {
                wi.z *= -1.0;
            }
            let pdf = cosine_hemisphere_pdf(abs_cos_theta(&wi)) * pr / (pr + pt);
            Some(BSDFSample::new(
                self.f(wo, &wi, mode),
                wi,
                pdf,
                BXDF_REFLECTION | BXDF_DIFFUSE,
                1.0,
                false,
            ))
        } else {
            // Sample diffuse BSDF transmission.
            let mut wi = cosine_sample_hemisphere(u);
            if wo.z > 0.0 {
                wi.z *= -1.0;
            }
            let pdf = cosine_hemisphere_pdf(abs_cos_theta(&wi)) * pt / (pr + pt);
            Some(BSDFSample::new(
                self.f(wo, &wi, mode),
                wi,
                pdf,
                BXDF_TRANSMISSION | BXDF_DIFFUSE,
                1.0,
                false,
            ))
        }
    }

    /// pbrt-v4 `DiffuseTransmissionBxDF::PDF`.
    pub fn pdf(
        &self,
        wo: &Vector3f,
        wi: &Vector3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        let mut pr = self.r.max_component_value();
        let mut pt = self.t.max_component_value();
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            pr = 0.0;
        }
        if sample_flags & BXDF_REFL_TRANS_TRANSMISSION == 0 {
            pt = 0.0;
        }
        if pr == 0.0 && pt == 0.0 {
            return 0.0;
        }

        if same_hemisphere(wo, wi) {
            pr / (pr + pt) * cosine_hemisphere_pdf(abs_cos_theta(wi))
        } else {
            pt / (pr + pt) * cosine_hemisphere_pdf(abs_cos_theta(wi))
        }
    }

    /// pbrt-v4 `DiffuseTransmissionBxDF::Flags`.
    pub fn flags(&self) -> BxDFFlags {
        let r_flag = if self.r.is_black() {
            BXDF_UNSET
        } else {
            BXDF_REFLECTION | BXDF_DIFFUSE
        };
        let t_flag = if self.t.is_black() {
            BXDF_UNSET
        } else {
            BXDF_TRANSMISSION | BXDF_DIFFUSE
        };
        r_flag | t_flag
    }

    /// pbrt-v4 `DiffuseTransmissionBxDF::Regularize` — no-op.
    pub fn regularize(&mut self) {}
}
