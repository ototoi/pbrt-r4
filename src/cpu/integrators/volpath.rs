use crate::base::bssrdf::SubsurfaceInteraction;
use crate::base::bxdf::{
    is_non_specular, is_reflective, is_transmissive, TransportMode, BXDF_ALL, BXDF_REFL_TRANS_ALL,
};
use crate::base::camera::Camera;
use crate::base::light::{is_delta_light, Light};
use crate::base::lightsampler::{LightSampleContext, LightSampler};
use crate::base::medium::{
    sample_t_maj_coefficients, sample_t_maj_sigma, MediumCoefficients, MediumSigma,
};
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
use crate::util::rng::RNG;
use crate::util::sampling::WeightedReservoirSampler;
use crate::util::spectrum::*;

/// pbrt-v4 `MixBits` (util/hash.h): permute a u64 so that nearby
/// inputs produce well-spread outputs. Used here just to derive a
/// per-event reservoir seed from the path RNG.
#[inline]
fn mix_bits(mut v: u64) -> u64 {
    v ^= v >> 31;
    v = v.wrapping_mul(0x7fb5d329728ea185);
    v ^= v >> 27;
    v = v.wrapping_mul(0x81dadef4bc2dd44d);
    v ^= v >> 33;
    v
}

use std::sync::Arc;
use std::sync::RwLock;

pub struct VolPathIntegrator {
    base: RayIntegratorBase,
    max_depth: i32,
    light_sampler: Option<LightSampler>,
    regularize: bool,
    light_sample_strategy: String,
}

impl VolPathIntegrator {
    pub fn new(
        max_depth: i32,
        scene: &Scene,
        camera: &Arc<Camera>,
        sampler: &Arc<RwLock<Sampler>>,
        pixel_bounds: &Bounds2i,
        light_sample_strategy: &str,
        regularize: bool,
    ) -> Self {
        VolPathIntegrator {
            base: RayIntegratorBase::new(scene, camera, sampler, pixel_bounds),
            max_depth,
            light_sampler: None,
            regularize,
            light_sample_strategy: light_sample_strategy.to_string(),
        }
    }

    /// pbrt-v4 `VolPathIntegrator::SampleLd(intr, bsdf, lambda, sampler,
    /// beta, r_p)` (integrators.cpp:1273-1390). Direct illumination
    /// with medium transmittance via ratio tracking. `bsdf == None`
    /// indicates a medium interaction (phase function); otherwise
    /// surface scattering.
    fn sample_ld(
        &self,
        intr: &Interaction,
        bsdf: Option<&BSDF>,
        lambda: &SampledWavelengths,
        sampler: &mut Sampler,
        beta: SampledSpectrum,
        r_p: SampledSpectrum,
    ) -> SampledSpectrum {
        let light_sampler = match self.light_sampler.as_ref() {
            Some(ls) => ls,
            None => return SampledSpectrum::zero(),
        };

        // Initialize LightSampleContext for volumetric light sampling
        let mut ctx;
        if let Some(bsdf) = bsdf {
            // Surface: use SurfaceInteraction's ctx with OffsetRayOrigin nudge.
            let si = match intr.as_surface_interaction() {
                Some(s) => s,
                None => return SampledSpectrum::zero(),
            };
            ctx = LightSampleContext::from(intr);
            let flags = bsdf.flags();
            if is_reflective(flags) && !is_transmissive(flags) {
                ctx.p = offset_ray_origin(&si.p, &si.p_error, &si.n, &si.wo);
            } else if is_transmissive(flags) && !is_reflective(flags) {
                ctx.p = offset_ray_origin(&si.p, &si.p_error, &si.n, &(-si.wo));
            }
        } else {
            // Medium interaction.
            ctx = LightSampleContext::from(intr);
        }

        // Sample a light source using lightSampler
        let u = sampler.get_1d();
        let sampled_light = light_sampler.sample(&ctx, u);
        let u_light = sampler.get_2d();
        let sampled_light = match sampled_light {
            Some(s) => s,
            None => return SampledSpectrum::zero(),
        };
        let light: &Arc<Light> = &sampled_light.light;
        if sampled_light.p == 0.0 {
            return SampledSpectrum::zero();
        }

        // Sample a point on the light source
        let ls = light.as_ref().sample_li(&ctx, u_light, lambda, true);
        let ls = match ls {
            Some(s) => s,
            None => return SampledSpectrum::zero(),
        };
        if ls.l.is_black() || ls.pdf == 0.0 {
            return SampledSpectrum::zero();
        }
        let p_l = sampled_light.p * ls.pdf;

        // Evaluate BSDF or phase function for light sample direction
        let wi = ls.wi;
        let (f_hat, scatter_pdf): (SampledSpectrum, Float);
        if let Some(bsdf) = bsdf {
            // Surface: f_hat = bsdf.f(wo, wi) * |wi · ns|
            let si = intr.as_surface_interaction().unwrap();
            let wo = si.wo;
            f_hat = bsdf.f(wo, wi, TransportMode::Radiance)
                * Float::abs(Vector3f::dot(&wi, &Vector3f::from(si.shading.n)));
            scatter_pdf = bsdf.pdf(wo, wi, TransportMode::Radiance, BXDF_REFL_TRANS_ALL);
        } else {
            // Medium: phase function. r4 stores phase on MediumInteraction.
            let mi = match intr.as_medium_interaction() {
                Some(m) => m,
                None => return SampledSpectrum::zero(),
            };
            let wo = mi.wo;
            let p_val = mi.phase.p(&wo, &wi);
            f_hat = SampledSpectrum::new(p_val);
            scatter_pdf = p_val;
        }
        if f_hat.is_black() {
            return SampledSpectrum::zero();
        }

        // Declare path state variables for ray to light source
        let p_light_intr = ls.p_light.clone();
        let mut light_ray = intr.spawn_ray_to(&p_light_intr);
        let mut t_ray = SampledSpectrum::one();
        let mut r_l = SampledSpectrum::one();
        let mut r_u = SampledSpectrum::one();
        // v4 `RNG rng(Hash(lightRay.o), Hash(lightRay.d))`.
        let mut rng = RNG::new();
        rng.set_sequence_with_seed(hash_point3f(&light_ray.o), hash_vec3f(&light_ray.d));

        while light_ray.d.length_squared() > 0.0 {
            // Trace ray through media to estimate transmittance
            let si = self.base.intersect(&light_ray, 1.0 - SHADOW_EPSILON);
            // Handle opaque surface along ray's path
            if let Some(s) = si.as_ref() {
                if s.intr.get_material().is_some() {
                    return SampledSpectrum::zero();
                }
            }

            // Update transmittance for current ray segment
            if let Some(medium) = light_ray.medium.as_ref() {
                let t_max = si.as_ref().map(|s| s.t_hit).unwrap_or(1.0 - SHADOW_EPSILON);
                let u = rng.uniform_float();

                use std::cell::Cell;
                let t_ray_cell = Cell::new(t_ray);
                let r_l_cell = Cell::new(r_l);
                let r_u_cell = Cell::new(r_u);

                let t_maj_final = sample_t_maj_sigma(
                    medium.as_ref(),
                    &light_ray,
                    t_max,
                    u,
                    lambda,
                    &mut rng,
                    |_p, mp: MediumSigma, sigma_maj, t_maj, rng| {
                        let sa = mp.sigma_a;
                        let ss = mp.sigma_s;
                        let sigma_n = clamp_zero(sigma_maj - sa - ss);
                        let pdf = t_maj[0] * sigma_maj[0];
                        if pdf <= 0.0 {
                            t_ray_cell.set(SampledSpectrum::zero());
                            return false;
                        }
                        let inv_pdf = 1.0 / pdf;
                        let t_ray_next = t_ray_cell.get() * t_maj * sigma_n * inv_pdf;
                        let r_l_next = r_l_cell.get() * t_maj * sigma_maj * inv_pdf;
                        let r_u_next = r_u_cell.get() * t_maj * sigma_n * inv_pdf;

                        let denom = (r_l_next + r_u_next).average();
                        let tr_val = if denom > 0.0 {
                            t_ray_next / denom
                        } else {
                            SampledSpectrum::zero()
                        };
                        let mut t_ray_final = t_ray_next;
                        if tr_val.max_component_value() < 0.05 {
                            let q: Float = 0.75;
                            if rng.uniform_float() < q {
                                t_ray_final = SampledSpectrum::zero();
                            } else {
                                t_ray_final = t_ray_final / (1.0 - q);
                            }
                        }
                        t_ray_cell.set(t_ray_final);
                        r_l_cell.set(r_l_next);
                        r_u_cell.set(r_u_next);
                        if t_ray_final.is_black() {
                            return false;
                        }
                        true
                    },
                );
                // Update transmittance estimate for final segment
                if t_maj_final[0] > 0.0 {
                    let inv_t_maj0 = 1.0 / t_maj_final[0];
                    t_ray = t_ray_cell.get() * t_maj_final * inv_t_maj0;
                    r_l = r_l_cell.get() * t_maj_final * inv_t_maj0;
                    r_u = r_u_cell.get() * t_maj_final * inv_t_maj0;
                } else {
                    t_ray = t_ray_cell.get();
                    r_l = r_l_cell.get();
                    r_u = r_u_cell.get();
                }
            }

            // Generate next ray segment or return final transmittance
            if t_ray.is_black() {
                return SampledSpectrum::zero();
            }
            match si {
                None => break,
                Some(s) => {
                    light_ray = Interaction::from(&s.intr).spawn_ray_to(&p_light_intr);
                }
            }
        }

        // Return path contribution function estimate for direct lighting
        r_l *= r_p * p_l;
        r_u *= r_p * scatter_pdf;
        if is_delta_light(light.as_ref().light_type()) {
            let denom = r_l.average();
            if denom > 0.0 {
                beta * f_hat * t_ray * ls.l / denom
            } else {
                SampledSpectrum::zero()
            }
        } else {
            let denom = (r_l + r_u).average();
            if denom > 0.0 {
                beta * f_hat * t_ray * ls.l / denom
            } else {
                SampledSpectrum::zero()
            }
        }
    }
}

impl Integrator for VolPathIntegrator {
    fn render(&mut self) {
        RayIntegratorBase::render(self);
    }
    fn get_camera(&self) -> Arc<Camera> {
        self.base.camera.clone()
    }
}

impl RayIntegrator for VolPathIntegrator {
    fn preprocess(&mut self, _sampler: &mut Sampler) {
        match LightSampler::create(&self.light_sample_strategy, &self.base.base) {
            Ok(ls) => self.light_sampler = Some(ls),
            Err(e) => log::warn!("VolPathIntegrator: {:?}", e),
        }
    }

    /// pbrt-v4 `VolPathIntegrator::Li` (integrators.cpp:953-1271).
    /// Line-by-line translation of the v4 volumetric path loop, including
    /// the BSSRDF probe and reservoir-sampling path.
    fn li(
        &self,
        r: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        _scratch_buffer: &mut MemoryArena,
        mut visible_surf: Option<&mut VisibleSurface>,
    ) -> SampledSpectrum {
        // State variables for volumetric path sampling
        let mut l = SampledSpectrum::zero();
        let mut beta = SampledSpectrum::one();
        let mut r_u = SampledSpectrum::one();
        let mut r_l = SampledSpectrum::one();
        let mut specular_bounce = false;
        let mut any_non_specular_bounces = false;
        let mut depth: i32 = 0;
        let mut eta_scale: Float = 1.0;
        let mut prev_intr_ctx = LightSampleContext::default();
        let mut ray = r.clone();

        let light_sampler = match self.light_sampler.as_ref() {
            Some(ls) => ls,
            None => return SampledSpectrum::zero(),
        };

        loop {
            // Sample segment of volumetric scattering path
            let si = self.base.intersect(&ray.ray, Float::INFINITY);
            if let Some(medium) = ray.ray.medium.clone() {
                // Sample the participating medium
                let t_max = si.as_ref().map(|s| s.t_hit).unwrap_or(Float::INFINITY);
                let hash0 = hash_f32(sampler.get_1d());
                let hash1 = hash_f32(sampler.get_1d());
                let mut rng = RNG::new();
                rng.set_sequence_with_seed(hash0, hash1);

                use std::cell::Cell;
                let beta_cell = Cell::new(beta);
                let l_cell = Cell::new(l);
                let r_u_cell = Cell::new(r_u);
                let r_l_cell = Cell::new(r_l);
                let depth_cell = Cell::new(depth);
                let specular_bounce_cell = Cell::new(specular_bounce);
                let any_nonspec_cell = Cell::new(any_non_specular_bounces);
                let scattered = Cell::new(false);
                let terminated = Cell::new(false);
                let new_ray_o = Cell::new(ray.ray.o);
                let new_ray_d = Cell::new(ray.ray.d);
                let new_prev_ctx = Cell::new(prev_intr_ctx);
                let max_depth = self.max_depth;

                let t_maj_final = sample_t_maj_coefficients(
                    medium.as_ref(),
                    &ray.ray,
                    t_max,
                    sampler.get_1d(),
                    &*lambda,
                    &mut rng,
                    |p, mp: MediumCoefficients, sigma_maj, t_maj, rng| {
                        if beta_cell.get().is_black() {
                            terminated.set(true);
                            return false;
                        }
                        let sa = mp.sigma_a;
                        let ss = mp.sigma_s;
                        let le_pkt = mp.le;

                        // Add emission from medium scattering event
                        if depth_cell.get() < max_depth && !le_pkt.is_black() {
                            let pdf = sigma_maj[0] * t_maj[0];
                            if pdf > 0.0 {
                                let beta_p = beta_cell.get() * t_maj / pdf;
                                let r_e = r_u_cell.get() * sigma_maj * t_maj / pdf;
                                let r_e_avg = r_e.average();
                                if r_e_avg > 0.0 {
                                    l_cell.set(l_cell.get() + beta_p * sa * le_pkt / r_e_avg);
                                }
                            }
                        }

                        // Compute medium event probabilities
                        if sigma_maj[0] <= 0.0 {
                            terminated.set(true);
                            return false;
                        }
                        let p_absorb = sa[0] / sigma_maj[0];
                        let p_scatter = ss[0] / sigma_maj[0];
                        let p_null = Float::max(0.0, 1.0 - p_absorb - p_scatter);
                        let um = rng.uniform_float();
                        let mode = sample_discrete3(&[p_absorb, p_scatter, p_null], um);
                        if mode == 0 {
                            // Absorption
                            terminated.set(true);
                            false
                        } else if mode == 1 {
                            // Scattering
                            let d_cur = depth_cell.get();
                            depth_cell.set(d_cur + 1);
                            if d_cur >= max_depth {
                                terminated.set(true);
                                return false;
                            }
                            let pdf = t_maj[0] * ss[0];
                            if pdf <= 0.0 {
                                terminated.set(true);
                                return false;
                            }
                            let inv_pdf = 1.0 / pdf;
                            beta_cell.set(beta_cell.get() * t_maj * ss * inv_pdf);
                            r_u_cell.set(r_u_cell.get() * t_maj * ss * inv_pdf);
                            if !beta_cell.get().is_black() && !r_u_cell.get().is_black() {
                                // Sample direct lighting at volume-scattering event.
                                // Build a MediumInteraction and call SampleLd.
                                // Pass `ray.medium` so the shadow ray spawned in
                                // SampleLd inherits it (v4 integrators.cpp:1027).
                                let phase = medium.sample_phase_function(&p, lambda);
                                let intr = MediumInteraction::new(
                                    &p,
                                    &-new_ray_d.get(),
                                    ray.ray.time,
                                    &ray.ray.medium,
                                    &phase,
                                );
                                let ld = self.sample_ld(
                                    &Interaction::Medium(intr.clone()),
                                    None,
                                    lambda,
                                    sampler,
                                    beta_cell.get(),
                                    r_u_cell.get(),
                                );
                                l_cell.set(l_cell.get() + ld);

                                // Sample new direction at real-scattering event
                                let u2 = sampler.get_2d();
                                let (ps_pdf, ps_wi) = intr.phase.sample_p(&-new_ray_d.get(), &u2);
                                if ps_pdf == 0.0 {
                                    terminated.set(true);
                                } else {
                                    let p_val = intr.phase.p(&-new_ray_d.get(), &ps_wi);
                                    beta_cell.set(beta_cell.get() * (p_val / ps_pdf));
                                    r_l_cell.set(r_u_cell.get() / ps_pdf);
                                    new_prev_ctx
                                        .set(LightSampleContext::from(&Interaction::Medium(intr)));
                                    scattered.set(true);
                                    new_ray_o.set(p);
                                    new_ray_d.set(ps_wi);
                                    specular_bounce_cell.set(false);
                                    any_nonspec_cell.set(true);
                                }
                            }
                            false
                        } else {
                            // Null scattering
                            let sigma_n = clamp_zero(sigma_maj - sa - ss);
                            let pdf = t_maj[0] * sigma_n[0];
                            if pdf <= 0.0 {
                                beta_cell.set(SampledSpectrum::zero());
                                terminated.set(true);
                                return false;
                            }
                            let inv_pdf = 1.0 / pdf;
                            beta_cell.set(beta_cell.get() * t_maj * sigma_n * inv_pdf);
                            r_u_cell.set(r_u_cell.get() * t_maj * sigma_n * inv_pdf);
                            r_l_cell.set(r_l_cell.get() * t_maj * sigma_maj * inv_pdf);
                            !beta_cell.get().is_black() && !r_u_cell.get().is_black()
                        }
                    },
                );

                beta = beta_cell.get();
                l = l_cell.get();
                r_u = r_u_cell.get();
                r_l = r_l_cell.get();
                depth = depth_cell.get();
                specular_bounce = specular_bounce_cell.get();
                any_non_specular_bounces = any_nonspec_cell.get();
                let term = terminated.get();
                let scat = scattered.get();
                if scat {
                    ray.ray.o = new_ray_o.get();
                    ray.ray.d = new_ray_d.get();
                    prev_intr_ctx = new_prev_ctx.get();
                }

                if term || beta.is_black() || r_u.is_black() {
                    return l;
                }
                if scat {
                    continue;
                }

                // Final segment T_maj normalization
                if t_maj_final[0] > 0.0 {
                    let inv = 1.0 / t_maj_final[0];
                    beta = beta * t_maj_final * inv;
                    r_u = r_u * t_maj_final * inv;
                    r_l = r_l * t_maj_final * inv;
                }
            }

            // Handle surviving unscattered rays
            let si = match si {
                Some(s) => s,
                None => {
                    // Accumulate contributions from infinite light sources
                    for light in self.base.infinite_lights.iter() {
                        let le = light.as_ref().le(&ray.ray, lambda);
                        if !le.is_black() {
                            if depth == 0 || specular_bounce {
                                let avg = r_u.average();
                                if avg > 0.0 {
                                    l += beta * le / avg;
                                }
                            } else {
                                let p_l = light_sampler.pmf(&prev_intr_ctx, light)
                                    * light.as_ref().pdf_li(&prev_intr_ctx, ray.ray.d, true);
                                let r_l_local = r_l * p_l;
                                let denom = (r_u + r_l_local).average();
                                if denom > 0.0 {
                                    l += beta * le / denom;
                                }
                            }
                        }
                    }
                    break;
                }
            };

            let mut isect = si.intr;
            let le = isect.le(-ray.ray.d, lambda);
            if !le.is_black() {
                if depth == 0 || specular_bounce {
                    let avg = r_u.average();
                    if avg > 0.0 {
                        l += beta * le / avg;
                    }
                } else if let Some(area_light) = isect.get_area_light() {
                    let p_l = light_sampler.pmf(&prev_intr_ctx, &area_light)
                        * area_light.as_ref().pdf_li(&prev_intr_ctx, ray.ray.d, true);
                    let r_l_local = r_l * p_l;
                    let denom = (r_u + r_l_local).average();
                    if denom > 0.0 {
                        l += beta * le / denom;
                    }
                }
            }

            // Get BSDF and skip over medium boundaries
            let mut bsdf = match isect.get_bsdf(
                &ray,
                self.base.camera.as_ref(),
                sampler.samples_per_pixel(),
                lambda,
                Some(sampler),
            ) {
                Some(b) => b,
                None => {
                    ray = isect.spawn_ray(&ray.ray.d).into();
                    continue;
                }
            };
            let bssrdf = isect.get_bssrdf(lambda);

            // Initialize visibleSurf at first intersection
            if depth == 0 {
                if let Some(vs) = visible_surf.as_deref_mut() {
                    let (uc_rho, u_rho) = rho_samples();
                    let albedo = bsdf.rho(isect.wo, &uc_rho, &u_rho);
                    *vs = VisibleSurface::new(&isect, albedo, lambda);
                }
            }

            // Terminate path if maximum depth reached
            let d_current = depth;
            depth += 1;
            if d_current >= self.max_depth {
                return l;
            }

            // pbrt-v4 regularizes after the first non-specular bounce and
            // before direct-light sampling at this vertex.
            if self.regularize && any_non_specular_bounces {
                bsdf.regularize();
            }

            // Sample illumination from lights
            if is_non_specular(bsdf.flags()) {
                let ld = self.sample_ld(
                    &Interaction::from(&isect),
                    Some(&bsdf),
                    lambda,
                    sampler,
                    beta,
                    r_u,
                );
                l += ld;
            }
            prev_intr_ctx = LightSampleContext::from(&Interaction::from(&isect));

            // Sample BSDF to get new volumetric path direction
            let wo = isect.wo;
            let u = sampler.get_1d();
            let bs = match bsdf.sample_f(wo, u, sampler.get_2d(), TransportMode::Radiance, BXDF_ALL)
            {
                Some(s) => s,
                None => break,
            };
            beta *= bs.f
                * (Float::abs(Vector3f::dot(&bs.wi, &Vector3f::from(isect.shading.n))) / bs.pdf);
            if bs.pdf_is_proportional {
                let actual_pdf = bsdf.pdf(wo, bs.wi, TransportMode::Radiance, BXDF_REFL_TRANS_ALL);
                r_l = if actual_pdf > 0.0 {
                    r_u / actual_pdf
                } else {
                    SampledSpectrum::zero()
                };
            } else {
                r_l = if bs.pdf > 0.0 {
                    r_u / bs.pdf
                } else {
                    SampledSpectrum::zero()
                };
            }
            specular_bounce = bs.is_specular();
            any_non_specular_bounces |= !bs.is_specular();
            if bs.is_transmission() {
                eta_scale *= bs.eta * bs.eta;
            }
            ray = isect.spawn_ray_with_differentials(&ray, &bsdf, &bs.wi, bs.flags, bs.eta);

            // pbrt-v4 BSSRDF probe segment + reservoir-sampled
            // subsurface scattering (integrators.cpp:1188-1254).
            if let Some(bssrdf) = bssrdf {
                if bs.is_transmission() {
                    // 1) Sample the probe segment.
                    let uc = sampler.get_1d();
                    let up = sampler.get_2d();
                    let Some(probe) = bssrdf.sample_sp(uc, &up, lambda) else {
                        break;
                    };

                    // 2) Walk the scene between probe.p0 -> probe.p1
                    //    collecting same-material intersections into a
                    //    weighted reservoir. Subsequent iterations
                    //    spawn from the latest hit via
                    //    `spawn_ray_to_point` so we don't immediately
                    //    self-intersect against the surface we just
                    //    landed on (matches v4 `base.SpawnRayTo(...)`).
                    let seed = mix_bits(sampler.get_1d().to_bits() as u64);
                    let mut reservoir: WeightedReservoirSampler<SubsurfaceInteraction> =
                        WeightedReservoirSampler::with_seed(seed);
                    let ref_material = isect.get_material();
                    let mut current_hit: Option<SurfaceInteraction> = None;
                    loop {
                        let (ray_probe, dir_len_sq) = match &current_hit {
                            None => {
                                // First iteration: shoot directly from
                                // probe.p0 toward probe.p1.
                                let dir = probe.p1 - probe.p0;
                                let d2 = dir.length_squared();
                                if d2 == 0.0 {
                                    break;
                                }
                                (Ray::new(&probe.p0, &dir, 1.0, ray.ray.time), d2)
                            }
                            Some(prev) => {
                                let r = prev.spawn_ray_to_point(&probe.p1);
                                let d2 = r.d.length_squared();
                                if d2 == 0.0 {
                                    break;
                                }
                                (r, d2)
                            }
                        };
                        let _ = dir_len_sq;
                        let Some(si_hit) = self.base.intersect(&ray_probe, 1.0) else {
                            break;
                        };
                        let hit = si_hit.intr.clone();
                        let hit_material = hit.get_material();
                        let same = match (&ref_material, &hit_material) {
                            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                            (None, None) => true,
                            _ => false,
                        };
                        if same {
                            reservoir.add(SubsurfaceInteraction::from_surface(&hit), 1.0);
                        }
                        current_hit = Some(hit);
                    }

                    if !reservoir.has_sample() {
                        break;
                    }
                    let sample_probability = reservoir.sample_probability();
                    let ssi = reservoir.take().expect("reservoir has a sample");

                    // 3) Resolve probe intersection into BSSRDF sample.
                    let Some(bsample) = bssrdf.probe_intersection_to_sample(&ssi, lambda) else {
                        break;
                    };
                    if bsample.sp.is_black() || bsample.pdf[0] <= 0.0 {
                        break;
                    }

                    // 4) Update path state with the BSSRDF sample.
                    let pdf_scalar = sample_probability * bsample.pdf[0];
                    if pdf_scalar <= 0.0 {
                        break;
                    }
                    beta *= bsample.sp / pdf_scalar;
                    r_u *= bsample.pdf / bsample.pdf[0];
                    let mut pi = ssi.to_surface();
                    pi.wo = bsample.wo;
                    prev_intr_ctx = LightSampleContext::from(&Interaction::Surface(pi.clone()));

                    any_non_specular_bounces = true;

                    // 5) Direct illumination at the probe-exit surface
                    //    through the Sw (NormalizedFresnel) BSDF.
                    let sw = bsample.sw.clone();
                    let ld = self.sample_ld(
                        &Interaction::Surface(pi.clone()),
                        Some(&sw),
                        lambda,
                        sampler,
                        beta,
                        r_u,
                    );
                    l += ld;

                    // 6) Sample new direction from Sw and continue
                    //    the path. v4 does NOT increment depth here
                    //    (the BSSRDF scatter is "free" relative to
                    //    the maxdepth budget).
                    let u_bsdf = sampler.get_1d();
                    let Some(bs2) = sw.sample_f(
                        bsample.wo,
                        u_bsdf,
                        sampler.get_2d(),
                        TransportMode::Radiance,
                        BXDF_ALL,
                    ) else {
                        break;
                    };
                    beta *= bs2.f
                        * (Float::abs(Vector3f::dot(&bs2.wi, &Vector3f::from(pi.shading.n)))
                            / bs2.pdf);
                    r_l = if bs2.pdf > 0.0 {
                        r_u / bs2.pdf
                    } else {
                        SampledSpectrum::zero()
                    };
                    specular_bounce = bs2.is_specular();
                    ray = pi.spawn_ray(&bs2.wi).into();
                }
            }

            // Possibly terminate volumetric path with Russian roulette
            if beta.is_black() {
                break;
            }
            let r_u_avg = r_u.average();
            if r_u_avg > 0.0 {
                let rr_beta = beta * eta_scale / r_u_avg;
                if rr_beta.max_component_value() < 1.0 && depth > 1 {
                    let q = Float::max(0.0, 1.0 - rr_beta.max_component_value());
                    if sampler.get_1d() < q {
                        break;
                    }
                    beta /= 1.0 - q;
                }
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

crate::impl_image_tile_integrator_via_ray!(VolPathIntegrator);

unsafe impl Sync for VolPathIntegrator {}

/// pbrt-v4 builds the rho samples inline; mirrored here for albedo
/// estimation at the first surface hit.
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

/// pbrt-v4 `SampleDiscrete({a, b, c}, u)` -> index.
fn sample_discrete3(weights: &[Float; 3], u: Float) -> i32 {
    let sum = weights[0] + weights[1] + weights[2];
    if sum <= 0.0 {
        return -1;
    }
    let mut cdf = weights[0] / sum;
    if u < cdf {
        return 0;
    }
    cdf += weights[1] / sum;
    if u < cdf {
        return 1;
    }
    2
}

/// pbrt-v4 `ClampZero(SampledSpectrum)`: component-wise max(s, 0).
fn clamp_zero(s: SampledSpectrum) -> SampledSpectrum {
    let mut out = s;
    for i in 0..N_SPECTRUM_SAMPLES {
        if out[i] < 0.0 {
            out[i] = 0.0;
        }
    }
    out
}

fn hash_f32(x: Float) -> u64 {
    let bits = x.to_bits() as u64;
    let mut z = bits.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn hash_point3f(p: &Point3f) -> u64 {
    hash_f32(p.x) ^ hash_f32(p.y).rotate_left(21) ^ hash_f32(p.z).rotate_left(43)
}

fn hash_vec3f(v: &Vector3f) -> u64 {
    hash_f32(v.x) ^ hash_f32(v.y).rotate_left(21) ^ hash_f32(v.z).rotate_left(43)
}

pub fn create_volpath_integrator(
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
    Ok(Arc::new(RwLock::new(VolPathIntegrator::new(
        max_depth,
        scene,
        camera,
        sampler,
        &pixel_bounds,
        &light_strategy,
        regularize,
    ))))
}
