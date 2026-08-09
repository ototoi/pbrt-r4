// pbrt-v4 verbatim translation of `class AOIntegrator`
// (integrators.h:295, integrators.cpp:1409).
//
// Ambient-occlusion integrator: sample a hemisphere direction, cast a
// visibility ray; on miss, return illumScale * illuminant(lambda) *
// cos(theta) / (Pi * pdf). v4 takes a single sample per pixel/path (the
// outer `RayIntegrator::EvaluatePixelSample` loop handles `samplesPerPixel`).

use crate::base::camera::Camera;
use crate::base::sampler::Sampler;
use crate::cpu::integrators::*;
use crate::film::VisibleSurface;
use crate::options::*;
use crate::paramdict::*;
use crate::scene::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::memory::*;
use crate::util::sampling::*;
use crate::util::spectrum::*;
use crate::util::vecmath::*;

use std::sync::Arc;
use std::sync::RwLock;

/// pbrt-v4 `class AOIntegrator : public RayIntegrator` (integrators.h:296).
pub struct AOIntegrator {
    base: RayIntegratorBase,
    cos_sample: bool,
    max_dist: Float,
    illuminant: Spectrum,
    illum_scale: Float,
}

impl AOIntegrator {
    /// pbrt-v4 `AOIntegrator::AOIntegrator(bool cosSample, Float maxDist,
    /// Camera, Sampler, Primitive, vector<Light>, Spectrum illuminant)`
    /// (integrators.cpp:1409).
    pub fn new(
        cos_sample: bool,
        max_dist: Float,
        illuminant: Spectrum,
        scene: &Scene,
        camera: &Arc<Camera>,
        sampler: &Arc<RwLock<Sampler>>,
        pixel_bounds: &Bounds2i,
    ) -> Self {
        let illum_scale = 1.0 / spectrum_to_photometric(&illuminant);
        AOIntegrator {
            base: RayIntegratorBase::new(scene, camera, sampler, pixel_bounds),
            cos_sample,
            max_dist,
            illuminant,
            illum_scale,
        }
    }
}

impl Integrator for AOIntegrator {
    fn render(&mut self) {
        RayIntegratorBase::render(self);
    }

    fn get_camera(&self) -> Arc<Camera> {
        self.base.camera.clone()
    }
}

impl RayIntegrator for AOIntegrator {
    /// pbrt-v4 `SampledSpectrum AOIntegrator::Li(RayDifferential ray,
    /// SampledWavelengths &lambda, Sampler sampler, ScratchBuffer
    /// &scratchBuffer, VisibleSurface *visibleSurface) const`
    /// (integrators.cpp:1418). Line-by-line translation; the `retry`
    /// loop becomes a `loop` block in Rust.
    fn li(
        &self,
        r: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        _scratch_buffer: &mut MemoryArena,
        _visible_surface: Option<&mut VisibleSurface>,
    ) -> SampledSpectrum {
        let mut ray = r.clone();
        // Intersect _ray_ with scene and store intersection in _isect_
        loop {
            let Some(mut si) = self.base.intersect(&ray.ray, Float::INFINITY) else {
                return SampledSpectrum::zero();
            };
            let Some(_bsdf) = si.intr.get_bsdf(
                &ray,
                self.base.camera.as_ref(),
                sampler.samples_per_pixel(),
                lambda,
                Some(sampler),
            ) else {
                // SkipIntersection: continue past this hit
                ray = RayDifferential::from(si.intr.spawn_ray(&ray.ray.d));
                continue;
            };

            // Compute coordinate frame based on true geometry, not shading geometry.
            let n = face_forward(&Vector3f::from(si.intr.n), &-ray.ray.d);

            let u = sampler.get_2d();
            let (wi_local, pdf) = if self.cos_sample {
                let wi = cosine_sample_hemisphere(&u);
                let pdf = cosine_hemisphere_pdf(wi.z.abs());
                (wi, pdf)
            } else {
                let wi = uniform_sample_hemisphere(&u);
                let pdf = uniform_hemisphere_pdf();
                (wi, pdf)
            };
            if pdf == 0.0 {
                return SampledSpectrum::zero();
            }

            let f = Frame::from_z(n);
            let wi = f.from_local(wi_local);

            // Divide by pi so that fully visible is one.
            let r2 = si.intr.spawn_ray(&wi);
            if !self.base.intersect_p(&r2, self.max_dist) {
                return self.illum_scale * self.illuminant.sample(lambda) * Vector3f::dot(&wi, &n)
                    / (PI * pdf);
            }
            return SampledSpectrum::zero();
        }
    }

    fn get_sampler(&self) -> Arc<RwLock<Sampler>> {
        Arc::clone(&self.base.sampler)
    }

    fn get_pixel_bounds(&self) -> Bounds2i {
        self.base.pixel_bounds
    }
}

crate::impl_image_tile_integrator_via_ray!(AOIntegrator);

unsafe impl Sync for AOIntegrator {}

pub fn create_ao_integrator(
    params: &ParameterDictionary,
    sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    let pixel_bounds = camera.get_film().read().unwrap().pixel_bounds();
    let cos_sample = params.get_one_bool("cossample", true);
    let max_dist = params.get_one_float("maxdistance", Float::INFINITY);
    let _ = PbrtOptions::get();

    let illuminant = lookup_named_spectrum("stdillum-D65").ok_or_else(|| {
        PbrtError::error("required named spectrum \"stdillum-D65\" is unavailable")
    })?;

    Ok(Arc::new(RwLock::new(AOIntegrator::new(
        cos_sample,
        max_dist,
        illuminant,
        scene,
        camera,
        sampler,
        &pixel_bounds,
    ))))
}
