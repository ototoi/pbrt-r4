use crate::base::bxdf::{
    is_non_specular, is_reflective, is_transmissive, TransportMode, BXDF_ALL, BXDF_REFL_TRANS_ALL,
};
use crate::base::camera::Camera;
use crate::base::light::{is_delta_light, Light};
use crate::base::lightsampler::{LightSampleContext, LightSampler};
use crate::base::sampler::Sampler;
use crate::bsdf::BSDF;
use crate::cpu::integrators::*;
use crate::film::VisibleSurface;
use crate::interaction::*;
use crate::paramdict::*;
use crate::scene::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::memory::*;
use crate::util::sampling::power_heuristic;
use crate::util::spectrum::*;
use crate::util::stats::*;

use std::sync::Arc;
use std::sync::RwLock;

thread_local!(static PATHS: StatPercent = StatPercent::new("Integrator/Zero-radiance paths"));
thread_local!(static PATH_LENGTH: StatIntDistribution = StatIntDistribution::new("Integrator/Path length"));

/// pbrt-v4 `class PathIntegrator` (integrators.h:207).
pub struct PathIntegrator {
    base: RayIntegratorBase,
    max_depth: i32,
    light_sampler: Option<LightSampler>,
    regularize: bool,
    light_sample_strategy: String,
}

/// pbrt-v4 `PathIntegrator::Li` (integrators.cpp:706) builds rho samples
/// inline. Mirror the same constants here for v4-identical albedo
/// estimation.
fn rho_samples() -> ([Float; 16], [Point2f; 16]) {
    (
        [
            0.75741637,
            0.37870818,
            0.7083487,
            0.18935409,
            0.9149363,
            0.35417435,
            0.5990858,
            0.09467703,
            0.8578725,
            0.45746812,
            0.686759,
            0.17708716,
            0.9674518,
            0.2995429,
            0.5083201,
            0.047338516,
        ],
        [
            Point2f::new(0.855985, 0.570367),
            Point2f::new(0.381823, 0.851844),
            Point2f::new(0.285328, 0.764262),
            Point2f::new(0.73338, 0.114073),
            Point2f::new(0.542663, 0.344465),
            Point2f::new(0.127274, 0.414848),
            Point2f::new(0.9647, 0.947162),
            Point2f::new(0.594089, 0.643463),
            Point2f::new(0.095109, 0.170369),
            Point2f::new(0.825444, 0.263359),
            Point2f::new(0.429467, 0.454469),
            Point2f::new(0.24446, 0.816459),
            Point2f::new(0.756135, 0.731258),
            Point2f::new(0.516165, 0.152852),
            Point2f::new(0.180888, 0.214174),
            Point2f::new(0.898579, 0.503897),
        ],
    )
}

impl PathIntegrator {
    pub fn new(
        max_depth: i32,
        scene: &Scene,
        camera: &Arc<Camera>,
        sampler: &Arc<RwLock<Sampler>>,
        pixel_bounds: &Bounds2i,
        light_sample_strategy: &str,
        regularize: bool,
    ) -> Self {
        PathIntegrator {
            base: RayIntegratorBase::new(scene, camera, sampler, pixel_bounds),
            max_depth,
            light_sampler: None,
            regularize,
            light_sample_strategy: light_sample_strategy.to_string(),
        }
    }

    /// pbrt-v4 `PathIntegrator::SampleLd(const SurfaceInteraction &intr,
    /// const BSDF *bsdf, SampledWavelengths &lambda, Sampler sampler)
    /// const` (integrators.cpp:764). Line-by-line translation.
    fn sample_ld(
        &self,
        intr: &SurfaceInteraction,
        bsdf: &BSDF,
        lambda: &SampledWavelengths,
        sampler: &mut Sampler,
    ) -> SampledSpectrum {
        let light_sampler = match self.light_sampler.as_ref() {
            Some(ls) => ls,
            None => return SampledSpectrum::zero(),
        };

        // Initialize LightSampleContext for light sampling
        let mut ctx = LightSampleContext::from(&Interaction::from(intr));
        let flags = bsdf.flags();
        if is_reflective(flags) && !is_transmissive(flags) {
            ctx.p = offset_ray_origin(&intr.p, &intr.p_error, &intr.n, &intr.wo);
        } else if is_transmissive(flags) && !is_reflective(flags) {
            ctx.p = offset_ray_origin(&intr.p, &intr.p_error, &intr.n, &(-intr.wo));
        }

        // Choose a light source for the direct lighting calculation
        let u = sampler.get_1d();
        let sampled_light = light_sampler.sample(&ctx, u);
        let u_light = sampler.get_2d();
        let Some(sampled_light) = sampled_light else {
            return SampledSpectrum::zero();
        };

        // Sample a point on the light source for direct lighting
        let light: &Arc<Light> = &sampled_light.light;
        let ls = light.as_ref().sample_li(&ctx, u_light, lambda, true);
        let Some(ls) = ls else {
            return SampledSpectrum::zero();
        };
        if ls.l.is_black() || ls.pdf == 0.0 {
            return SampledSpectrum::zero();
        }

        // Evaluate BSDF for light sample and check light visibility
        let wo = intr.wo;
        let wi = ls.wi;
        let f = bsdf.f(wo, wi, TransportMode::Radiance)
            * Float::abs(Vector3f::dot(&wi, &Vector3f::from(intr.shading.n)));
        if f.is_black() || !self.base.unoccluded(&Interaction::from(intr), &ls.p_light) {
            return SampledSpectrum::zero();
        }

        // Return light's contribution to reflected radiance
        let p_l = sampled_light.p * ls.pdf;
        if is_delta_light(light.as_ref().light_type()) {
            ls.l * f / p_l
        } else {
            let p_b = bsdf.pdf(wo, wi, TransportMode::Radiance, BXDF_REFL_TRANS_ALL);
            let w_l = power_heuristic(1, p_l, 1, p_b);
            w_l * ls.l * f / p_l
        }
    }
}

impl Integrator for PathIntegrator {
    fn render(&mut self) {
        RayIntegratorBase::render(self);
    }

    fn get_camera(&self) -> Arc<Camera> {
        self.base.camera.clone()
    }
}

impl RayIntegrator for PathIntegrator {
    fn preprocess(&mut self, _sampler: &mut Sampler) {
        match LightSampler::create(&self.light_sample_strategy, &self.base.base) {
            Ok(ls) => self.light_sampler = Some(ls),
            Err(e) => log::warn!("PathIntegrator: {:?}", e),
        }
    }

    /// pbrt-v4 `SampledSpectrum PathIntegrator::Li(RayDifferential ray,
    /// SampledWavelengths &lambda, Sampler sampler, ScratchBuffer
    /// &scratchBuffer, VisibleSurface *visibleSurf) const`
    /// (integrators.cpp:628). Line-by-line translation.
    fn li(
        &self,
        r: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        _scratch_buffer: &mut MemoryArena,
        mut visible_surf: Option<&mut VisibleSurface>,
    ) -> SampledSpectrum {
        // Declare local variables for PathIntegrator::Li()
        let mut l = SampledSpectrum::zero();
        let mut beta = SampledSpectrum::one();
        let mut depth: i32 = 0;

        let mut p_b: Float = 0.0;
        let mut eta_scale: Float = 1.0;
        let mut specular_bounce = false;
        let mut any_non_specular_bounces = false;
        let mut prev_intr_ctx = LightSampleContext::default();

        let mut ray = r.clone();

        let light_sampler = match self.light_sampler.as_ref() {
            Some(ls) => ls,
            None => return SampledSpectrum::zero(),
        };

        // Sample path from camera and accumulate radiance estimate
        loop {
            // Trace ray and find closest path vertex and its BSDF
            let si = self.base.intersect(&ray.ray, Float::INFINITY);
            // Add emitted light at intersection point or from the environment
            let Some(mut si) = si else {
                // Incorporate emission from infinite lights for escaped ray
                for light in self.base.infinite_lights.iter() {
                    let le = light.as_ref().le(&ray.ray, lambda);
                    if depth == 0 || specular_bounce {
                        l += beta * le;
                    } else {
                        // Compute MIS weight for infinite light
                        let p_l = light_sampler.pmf(&prev_intr_ctx, light)
                            * light.as_ref().pdf_li(&prev_intr_ctx, ray.ray.d, true);
                        let w_b = power_heuristic(1, p_b, 1, p_l);
                        l += beta * w_b * le;
                    }
                }
                break;
            };

            // Incorporate emission from surface hit by ray
            let le = si.intr.le(-ray.ray.d, lambda);
            if !le.is_black() {
                if depth == 0 || specular_bounce {
                    l += beta * le;
                } else if let Some(area_light) = si.intr.get_area_light() {
                    // Compute MIS weight for area light
                    let p_l = light_sampler.pmf(&prev_intr_ctx, &area_light)
                        * area_light.as_ref().pdf_li(&prev_intr_ctx, ray.ray.d, true);
                    let w_l = power_heuristic(1, p_b, 1, p_l);
                    l += beta * w_l * le;
                }
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
                    specular_bounce = true; // disable MIS if the indirect ray hits a light
                    ray = si.intr.spawn_ray(&ray.ray.d).into();
                    continue;
                }
            };

            // Initialize visibleSurf at first intersection
            if depth == 0 {
                if let Some(vs) = visible_surf.as_deref_mut() {
                    // Estimate BSDF's albedo using v4's fixed sample arrays.
                    let (uc_rho, u_rho) = rho_samples();
                    let albedo = bsdf.rho(si.intr.wo, &uc_rho, &u_rho);
                    *vs = VisibleSurface::new(&si.intr, albedo, lambda);
                }
            }

            let mut regularized_bsdf;
            let bsdf = if self.regularize && any_non_specular_bounces {
                regularized_bsdf = bsdf.clone();
                regularized_bsdf.regularize();
                &regularized_bsdf
            } else {
                &bsdf
            };

            // End path if maximum depth reached
            let d_current = depth;
            depth += 1;
            if d_current == self.max_depth {
                break;
            }

            // Sample direct illumination from the light sources
            if is_non_specular(bsdf.flags()) {
                PATHS.with(|s| s.add_denom(1));
                let ld = self.sample_ld(&si.intr, bsdf, lambda, sampler);
                if ld.is_black() {
                    PATHS.with(|s| s.add_num(1));
                }
                l += beta * ld;
            }

            // Sample BSDF to get new path direction
            let wo = -ray.ray.d;
            let u = sampler.get_1d();
            let Some(bs) =
                bsdf.sample_f(wo, u, sampler.get_2d(), TransportMode::Radiance, BXDF_ALL)
            else {
                break;
            };
            // Update path state variables after surface scattering
            beta *= bs.f
                * (Float::abs(Vector3f::dot(&bs.wi, &Vector3f::from(si.intr.shading.n))) / bs.pdf);
            p_b = if bs.pdf_is_proportional {
                bsdf.pdf(wo, bs.wi, TransportMode::Radiance, BXDF_REFL_TRANS_ALL)
            } else {
                bs.pdf
            };
            specular_bounce = bs.is_specular();
            any_non_specular_bounces |= !bs.is_specular();
            if bs.is_transmission() {
                eta_scale *= bs.eta * bs.eta;
            }
            prev_intr_ctx = LightSampleContext::from(&Interaction::from(&si.intr));

            ray = si
                .intr
                .spawn_ray_with_differentials(&ray, &bsdf, &bs.wi, bs.flags, bs.eta);

            // Possibly terminate the path with Russian roulette
            let rr_beta = beta * eta_scale;
            if rr_beta.max_component_value() < 1.0 && depth > 1 {
                let q = Float::max(0.0, 1.0 - rr_beta.max_component_value());
                if sampler.get_1d() < q {
                    break;
                }
                beta /= 1.0 - q;
            }
        }
        PATH_LENGTH.with(|s| s.add(depth as u64));
        l
    }

    fn get_sampler(&self) -> Arc<RwLock<Sampler>> {
        Arc::clone(&self.base.sampler)
    }

    fn get_pixel_bounds(&self) -> Bounds2i {
        self.base.pixel_bounds
    }
}

crate::impl_image_tile_integrator_via_ray!(PathIntegrator);

unsafe impl Sync for PathIntegrator {}

pub fn create_path_integrator(
    params: &ParameterDictionary,
    sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    let max_depth = params.get_one_int("maxdepth", 5);
    let film = camera.as_ref().get_film();
    let pixel_bounds = film.read().unwrap().pixel_bounds();
    let light_strategy = {
        let strategy = params.get_one_string("lightsampler", "");
        if strategy.is_empty() {
            params.get_one_string("lightsamplestrategy", "bvh")
        } else {
            strategy
        }
    };
    let regularize = params.get_one_bool("regularize", false);
    Ok(Arc::new(RwLock::new(PathIntegrator::new(
        max_depth,
        scene,
        camera,
        sampler,
        &pixel_bounds,
        &light_strategy,
        regularize,
    ))))
}
