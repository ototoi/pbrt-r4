use crate::bsdf::BSDF;
use crate::bssrdf::TabulatedBSSRDF;
use crate::cpu::integrators::IntegratorBase;
use crate::interaction::SurfaceInteraction;
use crate::util::base::*;
use crate::util::spectrum::*;

/// pbrt-v4 `BSSRDFProbeSegment` (bssrdf.h:95). A pair of world-space
/// endpoints that delimits where the BSSRDF probe ray should be
/// shot during subsurface sampling.
#[derive(Debug, Clone, Copy)]
pub struct BSSRDFProbeSegment {
    pub p0: Point3f,
    pub p1: Point3f,
}

impl BSSRDFProbeSegment {
    pub fn new(p0: Point3f, p1: Point3f) -> Self {
        Self { p0, p1 }
    }
}

/// pbrt-v4 `SubsurfaceInteraction` (bssrdf.h:32). A lightweight
/// snapshot of `SurfaceInteraction` carrying only the fields the
/// BSSRDF post-walk evaluation needs.
#[derive(Debug, Clone)]
pub struct SubsurfaceInteraction {
    pub p: Point3f,
    pub p_error: Vector3f,
    pub n: Normal3f,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub ns: Normal3f,
    pub dpdus: Vector3f,
    pub dpdvs: Vector3f,
    pub time: Float,
}

impl SubsurfaceInteraction {
    pub fn from_surface(si: &SurfaceInteraction) -> Self {
        Self {
            p: si.p,
            p_error: si.p_error,
            n: si.n,
            dpdu: si.dpdu,
            dpdv: si.dpdv,
            ns: si.shading.n,
            dpdus: si.shading.dpdu,
            dpdvs: si.shading.dpdv,
            time: si.time,
        }
    }

    pub fn to_surface(&self) -> SurfaceInteraction {
        let mut si = SurfaceInteraction::default();
        si.p = self.p;
        si.p_error = self.p_error;
        si.n = self.n;
        si.dpdu = self.dpdu;
        si.dpdv = self.dpdv;
        si.shading.n = self.ns;
        si.shading.dpdu = self.dpdus;
        si.shading.dpdv = self.dpdvs;
        si.time = self.time;
        si
    }
}

/// pbrt-v4 `BSSRDFSample` (bssrdf.h:25). The return value of
/// `BSSRDF::probe_intersection_to_sample`. `pdf` is per-wavelength
/// for proper rescaled-density MIS book-keeping inside `VolPathIntegrator`.
#[derive(Clone)]
pub struct BSSRDFSample {
    pub sp: SampledSpectrum,
    pub pdf: SampledSpectrum,
    pub sw: BSDF,
    pub wo: Vector3f,
}

impl std::fmt::Debug for BSSRDFSample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BSSRDFSample")
            .field("sp", &self.sp)
            .field("pdf", &self.pdf)
            .field("wo", &self.wo)
            .finish()
    }
}

#[derive(Debug)]
pub enum BSSRDF {
    Tabulated(TabulatedBSSRDF),
}

impl BSSRDF {
    /// Chain-based sample used by the BSSRDF sampling path.
    /// yet migrated to the v4-shape `sample_sp` / `probe_intersection_to_sample`
    /// split.
    pub fn sample_s(
        &self,
        base: &IntegratorBase,
        u1: Float,
        u2: &Point2f,
    ) -> Option<(SampledSpectrum, SurfaceInteraction, Float)> {
        match self {
            BSSRDF::Tabulated(bssrdf) => bssrdf.sample_s(base, u1, u2),
        }
    }

    /// pbrt-v4 `BSSRDF::SampleSp(u1, u2)` — produce just the world-space
    /// probe segment endpoints. Caller is expected to walk the scene
    /// between them and collect same-material intersections via
    /// `WeightedReservoirSampler`. `lambda` is required because v4
    /// uses `sigma_t[0]` (the hero wavelength of the current
    /// `SampledWavelengths`) to size the probe segment.
    pub fn sample_sp(
        &self,
        u1: Float,
        u2: &Point2f,
        lambda: &SampledWavelengths,
    ) -> Option<BSSRDFProbeSegment> {
        match self {
            BSSRDF::Tabulated(bssrdf) => bssrdf.sample_sp(u1, u2, lambda),
        }
    }

    /// pbrt-v4 `BSSRDF::ProbeIntersectionToSample(si, alloc)` — turn a
    /// chosen probe intersection into a full `BSSRDFSample` with
    /// per-wavelength PDF and a `NormalizedFresnelBxDF`-backed `BSDF`.
    pub fn probe_intersection_to_sample(
        &self,
        ssi: &SubsurfaceInteraction,
        lambda: &SampledWavelengths,
    ) -> Option<BSSRDFSample> {
        match self {
            BSSRDF::Tabulated(bssrdf) => bssrdf.probe_intersection_to_sample(ssi, lambda),
        }
    }
}
