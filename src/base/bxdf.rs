// BxDF enum-based implementation
// Ported from pbrt-v4's TaggedPtr-based BxDF design

use crate::bxdfs::*;
use crate::util::base::*;
use crate::util::sampling::{uniform_hemisphere_pdf, uniform_sample_hemisphere};
use crate::util::scattering::abs_cos_theta;
use crate::util::spectrum::*;

// BxDF flags
pub type BxDFFlags = u32;

pub const BXDF_UNSET: BxDFFlags = 0;
pub const BXDF_REFLECTION: BxDFFlags = 1 << 0; // 1
pub const BXDF_TRANSMISSION: BxDFFlags = 1 << 1; // 2
pub const BXDF_DIFFUSE: BxDFFlags = 1 << 2; // 4
pub const BXDF_GLOSSY: BxDFFlags = 1 << 3; // 8
pub const BXDF_SPECULAR: BxDFFlags = 1 << 4; // 16
pub const BXDF_ALL: BxDFFlags =
    BXDF_REFLECTION | BXDF_TRANSMISSION | BXDF_DIFFUSE | BXDF_GLOSSY | BXDF_SPECULAR;

// Helper functions for BxDFFlags
#[inline]
pub fn is_reflective(flags: BxDFFlags) -> bool {
    flags & BXDF_REFLECTION != 0
}

#[inline]
pub fn is_transmissive(flags: BxDFFlags) -> bool {
    flags & BXDF_TRANSMISSION != 0
}

#[inline]
pub fn is_diffuse(flags: BxDFFlags) -> bool {
    flags & BXDF_DIFFUSE != 0
}

#[inline]
pub fn is_glossy(flags: BxDFFlags) -> bool {
    flags & BXDF_GLOSSY != 0
}

#[inline]
pub fn is_specular(flags: BxDFFlags) -> bool {
    flags & BXDF_SPECULAR != 0
}

#[inline]
pub fn is_non_specular(flags: BxDFFlags) -> bool {
    flags & (BXDF_DIFFUSE | BXDF_GLOSSY) != 0
}

// BxDFReflTransFlags for sampling control
pub type BxDFReflTransFlags = u32;

pub const BXDF_REFL_TRANS_UNSET: BxDFReflTransFlags = 0;
pub const BXDF_REFL_TRANS_REFLECTION: BxDFReflTransFlags = 1 << 0;
pub const BXDF_REFL_TRANS_TRANSMISSION: BxDFReflTransFlags = 1 << 1;
pub const BXDF_REFL_TRANS_ALL: BxDFReflTransFlags =
    BXDF_REFL_TRANS_REFLECTION | BXDF_REFL_TRANS_TRANSMISSION;

// TransportMode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    Radiance,
    Importance,
}

impl std::ops::Not for TransportMode {
    type Output = TransportMode;
    fn not(self) -> Self::Output {
        match self {
            TransportMode::Radiance => TransportMode::Importance,
            TransportMode::Importance => TransportMode::Radiance,
        }
    }
}

/// Direct translation of pbrt-v4 `BSDFSample` (`base/bxdf.h:122`).
#[derive(Debug, Clone, Copy)]
pub struct BSDFSample {
    pub f: SampledSpectrum,
    pub wi: Vector3f,
    pub pdf: Float,
    pub flags: BxDFFlags,
    pub eta: Float,
    pub pdf_is_proportional: bool,
}

impl BSDFSample {
    pub fn new(
        f: SampledSpectrum,
        wi: Vector3f,
        pdf: Float,
        flags: BxDFFlags,
        eta: Float,
        pdf_is_proportional: bool,
    ) -> Self {
        BSDFSample {
            f,
            wi,
            pdf,
            flags,
            eta,
            pdf_is_proportional,
        }
    }

    #[inline]
    pub fn is_reflection(&self) -> bool {
        is_reflective(self.flags)
    }

    #[inline]
    pub fn is_transmission(&self) -> bool {
        is_transmissive(self.flags)
    }

    #[inline]
    pub fn is_diffuse(&self) -> bool {
        is_diffuse(self.flags)
    }

    #[inline]
    pub fn is_glossy(&self) -> bool {
        is_glossy(self.flags)
    }

    #[inline]
    pub fn is_specular(&self) -> bool {
        is_specular(self.flags)
    }
}

// BxDF enum - tagged union of all BxDF types
// Ported from pbrt-v4's TaggedPtr<...> implementation
#[derive(Clone)]
pub enum BxDF {
    CoatedConductor(Box<CoatedConductorBxDF>),
    CoatedDiffuse(Box<CoatedDiffuseBxDF>),
    Diffuse(Box<DiffuseBxDF>),
    Measured(Box<MeasuredBxDF>),
    DiffuseTransmission(Box<DiffuseTransmissionBxDF>),
    Hair(Box<HairBxDF>),
    SpecularReflection(Box<SpecularReflectionBxDF>),
    NormalizedFresnel(Box<NormalizedFresnelBxDF>),
    Dielectric(Box<DielectricBxDF>),
    ThinDielectric(Box<ThinDielectricBxDF>),
    Conductor(Box<ConductorBxDF>),
}

impl BxDF {
    /// pbrt-v4 `BxDF::f(wo, wi, mode)` — returns a `SampledSpectrum`
    /// evaluated at the wavelengths the BxDF was built for. Each
    /// concrete `BxDF` stores its reflectance / transmittance / etc.
    /// as `SampledSpectrum` set up at `Material::get_bxdf` time, so
    /// no `lambda` argument is needed here.
    pub fn f(&self, wo: &Vector3f, wi: &Vector3f, mode: TransportMode) -> SampledSpectrum {
        match self {
            BxDF::CoatedConductor(bxdf) => bxdf.f(wo, wi, mode),
            BxDF::CoatedDiffuse(bxdf) => bxdf.f(wo, wi, mode),
            BxDF::Diffuse(bxdf) => bxdf.f(wo, wi, mode),
            BxDF::Measured(bxdf) => bxdf.f(wo, wi, mode),
            BxDF::DiffuseTransmission(bxdf) => bxdf.f(wo, wi, mode),
            BxDF::Hair(bxdf) => bxdf.f(wo, wi, mode),
            BxDF::SpecularReflection(bxdf) => bxdf.f(wo, wi, mode),
            BxDF::NormalizedFresnel(bxdf) => bxdf.f(wo, wi, mode),
            BxDF::Dielectric(bxdf) => bxdf.f(wo, wi, mode),
            BxDF::ThinDielectric(bxdf) => bxdf.f(wo, wi, mode),
            BxDF::Conductor(bxdf) => bxdf.f(wo, wi, mode),
        }
    }

    /// Sample the BxDF
    pub fn sample_f(
        &self,
        wo: &Vector3f,
        uc: Float,
        u: &Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        match self {
            BxDF::CoatedConductor(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::CoatedDiffuse(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::Diffuse(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::Measured(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::DiffuseTransmission(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::Hair(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::SpecularReflection(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::NormalizedFresnel(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::Dielectric(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::ThinDielectric(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::Conductor(bxdf) => bxdf.sample_f(wo, uc, u, mode, sample_flags),
        }
    }

    /// Compute the PDF for the given directions
    pub fn pdf(
        &self,
        wo: &Vector3f,
        wi: &Vector3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        match self {
            BxDF::CoatedConductor(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
            BxDF::CoatedDiffuse(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
            BxDF::Diffuse(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
            BxDF::Measured(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
            BxDF::DiffuseTransmission(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
            BxDF::Hair(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
            BxDF::SpecularReflection(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
            BxDF::NormalizedFresnel(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
            BxDF::Dielectric(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
            BxDF::ThinDielectric(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
            BxDF::Conductor(bxdf) => bxdf.pdf(wo, wi, mode, sample_flags),
        }
    }

    /// Get the flags for this BxDF
    pub fn flags(&self) -> BxDFFlags {
        match self {
            BxDF::CoatedConductor(bxdf) => bxdf.flags(),
            BxDF::CoatedDiffuse(bxdf) => bxdf.flags(),
            BxDF::Diffuse(bxdf) => bxdf.flags(),
            BxDF::Measured(bxdf) => bxdf.flags(),
            BxDF::DiffuseTransmission(bxdf) => bxdf.flags(),
            BxDF::Hair(bxdf) => bxdf.flags(),
            BxDF::SpecularReflection(bxdf) => bxdf.flags(),
            BxDF::NormalizedFresnel(bxdf) => bxdf.flags(),
            BxDF::Dielectric(bxdf) => bxdf.flags(),
            BxDF::ThinDielectric(bxdf) => bxdf.flags(),
            BxDF::Conductor(bxdf) => bxdf.flags(),
        }
    }

    /// Regularize the BxDF (for variance reduction)
    pub fn regularize(&mut self) {
        match self {
            BxDF::CoatedConductor(bxdf) => {
                bxdf.regularize();
            }
            BxDF::CoatedDiffuse(bxdf) => {
                bxdf.regularize();
            }
            BxDF::Diffuse(bxdf) => {
                bxdf.regularize();
            }
            BxDF::Measured(bxdf) => {
                bxdf.regularize();
            }
            BxDF::DiffuseTransmission(bxdf) => {
                bxdf.regularize();
            }
            BxDF::Hair(bxdf) => {
                bxdf.regularize();
            }
            BxDF::SpecularReflection(bxdf) => {
                bxdf.regularize();
            }
            BxDF::NormalizedFresnel(bxdf) => {
                bxdf.regularize();
            }
            BxDF::Dielectric(bxdf) => {
                bxdf.regularize();
            }
            BxDF::ThinDielectric(bxdf) => {
                bxdf.regularize();
            }
            BxDF::Conductor(bxdf) => {
                bxdf.regularize();
            }
        }
    }

    /// pbrt-v4 `BxDF::rho(wo, uc, u)` (`base/bxdf.h:209`).
    pub fn rho(&self, wo: &Vector3f, uc: &[Float], u: &[Point2f]) -> SampledSpectrum {
        if uc.is_empty() || uc.len() != u.len() {
            return SampledSpectrum::zero();
        }

        let mut sum = SampledSpectrum::zero();
        for (uc_i, u_i) in uc.iter().zip(u.iter()) {
            if let Some(bs) =
                self.sample_f(wo, *uc_i, u_i, TransportMode::Radiance, BXDF_REFL_TRANS_ALL)
            {
                if bs.pdf > 0.0 {
                    sum += bs.f * (abs_cos_theta(&bs.wi) / bs.pdf);
                }
            }
        }

        sum / uc.len() as Float
    }

    /// pbrt-v4 `BxDF::rho(u1, uc, u2)` hemispherical-hemispherical
    /// reflectance (`base/bxdf.h:223`).
    pub fn rho2(&self, uc: &[Point2f], u: &[Point2f]) -> SampledSpectrum {
        if uc.is_empty() || uc.len() != u.len() {
            return SampledSpectrum::zero();
        }

        let mut sum = SampledSpectrum::zero();
        for (uc_i, u_i) in uc.iter().zip(u.iter()) {
            let wo = uniform_sample_hemisphere(uc_i);
            let pdf_wo = uniform_hemisphere_pdf();
            if let Some(bs) = self.sample_f(
                &wo,
                u_i.x,
                u_i,
                TransportMode::Radiance,
                BXDF_REFL_TRANS_ALL,
            ) {
                if bs.pdf > 0.0 {
                    sum += bs.f * (abs_cos_theta(&bs.wi) * abs_cos_theta(&wo) / (pdf_wo * bs.pdf));
                }
            }
        }

        sum / (PI * uc.len() as Float)
    }
}

impl std::fmt::Debug for BxDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BxDF::CoatedConductor(_bxdf) => write!(f, "BxDF::CoatedConductor"),
            BxDF::CoatedDiffuse(_bxdf) => write!(f, "BxDF::CoatedDiffuse"),
            BxDF::Diffuse(_bxdf) => write!(f, "BxDF::Diffuse"),
            BxDF::Measured(bxdf) => write!(f, "BxDF::Measured({})", bxdf.data().filename),
            BxDF::DiffuseTransmission(bxdf) => {
                write!(
                    f,
                    "BxDF::DiffuseTransmission(r={:?}, t={:?})",
                    bxdf.r, bxdf.t
                )
            }
            BxDF::Hair(_bxdf) => write!(f, "BxDF::Hair"),
            BxDF::SpecularReflection(_bxdf) => write!(f, "BxDF::SpecularReflection"),
            BxDF::NormalizedFresnel(_bxdf) => write!(f, "BxDF::NormalizedFresnel"),
            BxDF::Dielectric(_bxdf) => write!(f, "BxDF::Dielectric"),
            BxDF::ThinDielectric(bxdf) => write!(f, "BxDF::ThinDielectric(eta={})", bxdf.eta()),
            BxDF::Conductor(_bxdf) => write!(f, "BxDF::Conductor"),
        }
    }
}
