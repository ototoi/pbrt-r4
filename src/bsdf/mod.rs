// BSDF - Bidirectional Scattering Distribution Function
// Ported from pbrt-v4's BSDF class
//
// The BSDF wraps a BxDF and handles coordinate frame transformations
// between rendering space and local shading space.

use crate::base::bxdf::*;
use crate::util::base::*;
use crate::util::spectrum::*;
use crate::util::vecmath::*;

/// BSDF - combines a BxDF with a shading coordinate frame
#[derive(Clone)]
pub struct BSDF {
    pub bxdf: BxDF,
    pub shading_frame: Frame,
}

impl BSDF {
    /// Create a new BSDF with the given shading normal, tangent, and BxDF
    pub fn new(ns: Normal3f, dpdus: Vector3f, bxdf: BxDF) -> Self {
        let shading_frame = Frame::from_xz(dpdus.normalize(), Vector3f::from(ns));
        BSDF {
            bxdf,
            shading_frame,
        }
    }

    /// Get the BxDF flags
    pub fn flags(&self) -> BxDFFlags {
        self.bxdf.flags()
    }

    /// Transform vector from render space to local space
    pub fn render_to_local(&self, v: Vector3f) -> Vector3f {
        self.shading_frame.to_local(v)
    }

    /// Transform vector from local space to render space
    pub fn local_to_render(&self, v: Vector3f) -> Vector3f {
        self.shading_frame.from_local(v)
    }

    /// pbrt-v4 `BSDF::f(woRender, wiRender, mode)`
    /// (`base/bsdf.h:67`). Returns the BSDF response as a
    /// `SampledSpectrum` evaluated at the wavelengths the BxDF was
    /// constructed for.
    pub fn f(
        &self,
        wo_render: Vector3f,
        wi_render: Vector3f,
        mode: TransportMode,
    ) -> SampledSpectrum {
        let wo = self.render_to_local(wo_render);
        let wi = self.render_to_local(wi_render);

        if wo.z == 0.0 {
            return SampledSpectrum::zero();
        }

        self.bxdf.f(&wo, &wi, mode)
    }

    /// pbrt-v4 `BSDF::Sample_f(woRender, uc, u, mode, sampleFlags)`
    /// (`base/bsdf.h:73`).
    pub fn sample_f(
        &self,
        wo_render: Vector3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        let wo = self.render_to_local(wo_render);

        if wo.z == 0.0 {
            return None;
        }

        if (self.bxdf.flags() & sample_flags) == 0 {
            return None;
        }

        let mut bs = self.bxdf.sample_f(&wo, uc, &u, mode, sample_flags)?;

        if bs.f.is_black() || bs.pdf == 0.0 || bs.wi.z == 0.0 {
            return None;
        }

        // Transform wi back to render space
        bs.wi = self.local_to_render(bs.wi);

        Some(bs)
    }

    /// Compute the PDF for the given directions
    pub fn pdf(
        &self,
        wo_render: Vector3f,
        wi_render: Vector3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        let wo = self.render_to_local(wo_render);
        let wi = self.render_to_local(wi_render);

        if wo.z == 0.0 {
            return 0.0;
        }

        self.bxdf.pdf(&wo, &wi, mode, sample_flags)
    }

    /// Regularize the BSDF
    pub fn regularize(&mut self) {
        self.bxdf.regularize();
    }

    /// pbrt-v4 `BSDF::rho(woRender, uc, u)` (`base/bsdf.h:118`).
    pub fn rho(&self, wo_render: Vector3f, uc: &[Float], u: &[Point2f]) -> SampledSpectrum {
        let wo = self.render_to_local(wo_render);
        if wo.z == 0.0 {
            return SampledSpectrum::zero();
        }

        self.bxdf.rho(&wo, uc, u)
    }

    /// Get the number of BxDF components matching the given flags.
    pub fn num_components(&self, flags: BxDFFlags) -> usize {
        if (self.bxdf.flags() & flags) != 0 {
            1
        } else {
            0
        }
    }

    /// Check if BSDF has components matching the given flags
    pub fn has_components(&self, flags: BxDFFlags) -> bool {
        self.num_components(flags) > 0
    }
}

impl std::fmt::Debug for BSDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BSDF").field("bxdf", &self.bxdf).finish()
    }
}
