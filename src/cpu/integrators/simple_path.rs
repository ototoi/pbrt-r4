// pbrt-v4 verbatim translation of `class SimplePathIntegrator`
// (integrators.h:182, integrators.cpp:379). No MIS, no Russian roulette,
// no BSSRDF; just optional direct-light sampling and optional BSDF
// importance sampling toggled by user parameters.

use crate::base::bxdf::TransportMode;
use crate::base::bxdf::{
    is_reflective, is_transmissive, BxDFReflTransFlags, BXDF_ALL, BXDF_REFL_TRANS_ALL,
};
use crate::base::camera::Camera;
use crate::base::lightsampler::{LightSampleContext, UniformLightSampler};
use crate::base::sampler::Sampler;
use crate::cpu::integrators::*;
use crate::film::VisibleSurface;
use crate::interaction::Interaction;
use crate::paramdict::*;
use crate::scene::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::memory::*;
use crate::util::sampling::*;
use crate::util::spectrum::*;

use std::sync::Arc;
use std::sync::RwLock;

/// pbrt-v4 `class SimplePathIntegrator` (integrators.h:183).
pub struct SimplePathIntegrator {
    base: RayIntegratorBase,
    max_depth: i32,
    sample_lights: bool,
    sample_bsdf: bool,
    light_sampler: UniformLightSampler,
}

impl SimplePathIntegrator {
    /// pbrt-v4 `SimplePathIntegrator::SimplePathIntegrator(int maxDepth,
    /// bool sampleLights, bool sampleBSDF, Camera, Sampler, Primitive,
    /// vector<Light>)` (integrators.cpp:379).
    pub fn new(
        max_depth: i32,
        sample_lights: bool,
        sample_bsdf: bool,
        scene: &Scene,
        camera: &Arc<Camera>,
        sampler: &Arc<RwLock<Sampler>>,
        pixel_bounds: &Bounds2i,
    ) -> Self {
        let base = RayIntegratorBase::new(scene, camera, sampler, pixel_bounds);
        let light_sampler = UniformLightSampler::new(&base.base.base);
        SimplePathIntegrator {
            base,
            max_depth: max_depth.max(0),
            sample_lights,
            sample_bsdf,
            light_sampler,
        }
    }
}

impl Integrator for SimplePathIntegrator {
    fn render(&mut self) {
        RayIntegratorBase::render(self);
    }

    fn get_camera(&self) -> Arc<Camera> {
        self.base.camera.clone()
    }
}

impl RayIntegrator for SimplePathIntegrator {
    /// pbrt-v4 `SampledSpectrum SimplePathIntegrator::Li(RayDifferential
    /// ray, SampledWavelengths &lambda, Sampler sampler, ScratchBuffer
    /// &scratchBuffer, VisibleSurface *) const` (integrators.cpp:389).
    /// Line-by-line translation.
    fn li(
        &self,
        r: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        _scratch_buffer: &mut MemoryArena,
        _visible_surface: Option<&mut VisibleSurface>,
    ) -> SampledSpectrum {
        // Estimate radiance along ray using simple path tracing
        let mut l = SampledSpectrum::zero();
        let mut beta = SampledSpectrum::one();
        let mut specular_bounce = true;
        let mut depth: i32 = 0;
        let mut ray = r.clone();

        while !beta.is_black() {
            // Find next vertex and accumulate contribution.
            // Intersect ray with scene
            let si = self.base.intersect(&ray.ray, Float::INFINITY);

            // Account for infinite lights if ray has no intersection
            let Some(mut si) = si else {
                if !self.sample_lights || specular_bounce {
                    for light in self.base.infinite_lights.iter() {
                        l += beta * light.as_ref().le(&ray.ray, lambda);
                    }
                }
                break;
            };

            // Account for emissive surface if light was not sampled
            if !self.sample_lights || specular_bounce {
                l += beta * si.intr.le(-ray.ray.d, lambda);
            }

            // End path if maximum depth reached
            let d = depth;
            depth += 1;
            if d == self.max_depth {
                break;
            }

            // Get BSDF and skip over medium boundaries
            let bsdf = match si.intr.get_bsdf(
                &ray,
                self.base.camera.as_ref(),
                sampler.samples_per_pixel(),
                lambda,
                Some(sampler),
            ) {
                Some(b) => b,
                None => {
                    specular_bounce = true;
                    ray = si.intr.spawn_ray(&ray.ray.d).into();
                    continue;
                }
            };

            // Sample direct illumination if sampleLights is true
            let wo = -ray.ray.d;
            if self.sample_lights {
                if let Some(sampled_light) = self.light_sampler.sample(sampler.get_1d()) {
                    // Sample point on sampled_light to estimate direct illumination
                    let u_light = sampler.get_2d();
                    let ctx = LightSampleContext::from(&Interaction::from(&si.intr));
                    if let Some(ls) = sampled_light.light.sample_li(&ctx, u_light, lambda, false) {
                        if !ls.l.is_black() && ls.pdf > 0.0 {
                            // Evaluate BSDF for light and possibly add scattered radiance
                            let wi = ls.wi;
                            let f = bsdf.f(wo, wi, TransportMode::Radiance)
                                * Float::abs(Vector3f::dot(
                                    &wi,
                                    &Vector3f::from(si.intr.shading.n),
                                ));
                            if !f.is_black()
                                && self
                                    .base
                                    .unoccluded(&Interaction::from(&si.intr), &ls.p_light)
                            {
                                l += beta * f * ls.l / (sampled_light.p * ls.pdf);
                            }
                        }
                    }
                }
            }

            // Sample outgoing direction at intersection to continue path
            if self.sample_bsdf {
                // Sample BSDF for new path direction
                let u = sampler.get_1d();
                let Some(bs) =
                    bsdf.sample_f(wo, u, sampler.get_2d(), TransportMode::Radiance, BXDF_ALL)
                else {
                    break;
                };
                beta *= bs.f
                    * (Float::abs(Vector3f::dot(&bs.wi, &Vector3f::from(si.intr.shading.n)))
                        / bs.pdf);
                specular_bounce = bs.is_specular();
                ray = si.intr.spawn_ray(&bs.wi).into();
            } else {
                // Uniformly sample sphere or hemisphere to get new path direction
                let flags = bsdf.flags();
                let (wi, pdf) = if is_reflective(flags) && is_transmissive(flags) {
                    let wi = uniform_sample_sphere(&sampler.get_2d());
                    (wi, uniform_sphere_pdf())
                } else {
                    let mut wi = uniform_sample_hemisphere(&sampler.get_2d());
                    let pdf = uniform_hemisphere_pdf();
                    let dot_wo_n = Vector3f::dot(&wo, &Vector3f::from(si.intr.n));
                    let dot_wi_n = Vector3f::dot(&wi, &Vector3f::from(si.intr.n));
                    if is_reflective(flags) && dot_wo_n * dot_wi_n < 0.0 {
                        wi = -wi;
                    } else if is_transmissive(flags) && dot_wo_n * dot_wi_n > 0.0 {
                        wi = -wi;
                    }
                    (wi, pdf)
                };
                let f = bsdf.f(wo, wi, TransportMode::Radiance);
                if f.is_black() || pdf == 0.0 {
                    break;
                }
                beta *=
                    f * (Float::abs(Vector3f::dot(&wi, &Vector3f::from(si.intr.shading.n))) / pdf);
                specular_bounce = false;
                ray = si.intr.spawn_ray(&wi).into();
            }
        }

        l
    }

    fn get_sampler(&self) -> Arc<RwLock<Sampler>> {
        Arc::clone(&self.base.sampler)
    }

    fn get_pixel_bounds(&self) -> Bounds2i {
        self.base.pixel_bounds
    }
}

crate::impl_image_tile_integrator_via_ray!(SimplePathIntegrator);

unsafe impl Sync for SimplePathIntegrator {}

pub fn create_simple_path_integrator(
    params: &ParameterDictionary,
    sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    let pixel_bounds = camera.get_film().read().unwrap().pixel_bounds();
    let max_depth = params.get_one_int("maxdepth", 5);
    let sample_lights = params.get_one_bool("samplelights", true);
    let sample_bsdf = params.get_one_bool("samplebsdf", true);
    let _ = BxDFReflTransFlags::from(BXDF_REFL_TRANS_ALL);
    Ok(Arc::new(RwLock::new(SimplePathIntegrator::new(
        max_depth,
        sample_lights,
        sample_bsdf,
        scene,
        camera,
        sampler,
        &pixel_bounds,
    ))))
}
