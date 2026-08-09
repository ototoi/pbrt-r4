// pbrt-v4 verbatim translation of `class VisibleSurface` (film.h:133-160).
// `albedo` is a `SampledSpectrum` evaluated at the wavelength packet that
// produced this surface; storing the dense `Spectrum` here would lose
// information.

use crate::interaction::SurfaceInteraction;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::spectrum::*;
use crate::util::transform::AnimatedTransform;

#[derive(Debug, Clone)]
pub struct VisibleSurface {
    pub p: Point3f,
    pub n: Normal3f,
    pub ns: Normal3f,
    pub uv: Point2f,
    pub time: Float,
    pub dpdx: Vector3f,
    pub dpdy: Vector3f,
    pub albedo: SampledSpectrum,
    pub set: bool,
}

impl Default for VisibleSurface {
    fn default() -> Self {
        Self {
            p: Point3f::zero(),
            n: Normal3f::zero(),
            ns: Normal3f::zero(),
            uv: Point2f::zero(),
            time: 0.0,
            dpdx: Vector3f::zero(),
            dpdy: Vector3f::zero(),
            albedo: SampledSpectrum::zero(),
            set: false,
        }
    }
}

impl VisibleSurface {
    /// pbrt-v4 `VisibleSurface(const SurfaceInteraction &si,
    /// SampledSpectrum albedo, const SampledWavelengths &lambda)`
    /// (film.cpp).
    pub fn new(
        si: &SurfaceInteraction,
        albedo: SampledSpectrum,
        _lambda: &SampledWavelengths,
    ) -> Self {
        let wo = si.wo;
        let n = Normal3f::from(face_forward(&Vector3f::from(si.n), &wo));
        let ns = Normal3f::from(face_forward(&Vector3f::from(si.shading.n), &wo));
        Self {
            p: si.p,
            n,
            ns,
            uv: si.uv,
            time: si.time,
            dpdx: si.dpdx,
            dpdy: si.dpdy,
            albedo,
            set: true,
        }
    }

    pub fn transformed(&self, transform: &AnimatedTransform) -> Self {
        if !self.set {
            return self.clone();
        }

        let p = transform.transform_point(self.time, &self.p);
        let n = transform.transform_normal(self.time, &self.n).normalize();
        let ns = transform.transform_normal(self.time, &self.ns).normalize();
        let dpdx = transform.transform_vector(self.time, &self.dpdx);
        let dpdy = transform.transform_vector(self.time, &self.dpdy);

        Self {
            p,
            n,
            ns,
            uv: self.uv,
            time: self.time,
            dpdx,
            dpdy,
            albedo: self.albedo,
            set: true,
        }
    }
}
