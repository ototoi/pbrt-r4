// CPU SPPM implementation.

use crate::base::bxdf::{
    is_diffuse, is_glossy, is_non_specular, is_reflective, is_transmissive, TransportMode,
    BXDF_ALL, BXDF_REFL_TRANS_ALL,
};
use crate::base::camera::Camera;
use crate::base::light::{is_delta_light, Light};
use crate::base::lightsampler::{
    BVHLightSampler, LightSampleContext, LightSampler, PowerLightSampler, UniformLightSampler,
};
use crate::base::sampler::Sampler;
use crate::bsdf::BSDF;
use crate::cpu::integrators::*;
use crate::film::Film;
use crate::interaction::*;
use crate::options::PbrtOptions;
use crate::paramdict::*;
use crate::samplers::halton::permutation_for_dimension;
use crate::scene::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::lowdiscrepancy::radical_inverse::{
    compute_radical_inverse_permutations, radical_inverse, scrambled_radical_inverse,
};
use crate::util::memory::*;
use crate::util::rng::RNG;
use crate::util::sampling::power_heuristic;
use crate::util::spectrum::*;

use rayon::prelude::*;
use std::sync::{Arc, Mutex, RwLock};

pub struct SPPMIntegrator {
    pub base: IntegratorBase,
    pub camera: Arc<Camera>,
    pub sampler_prototype: Arc<RwLock<Sampler>>,
    pub initial_search_radius: Float,
    pub max_depth: i32,
    pub photons_per_iteration: i32,
    pub seed: i32,
}

/// pbrt-v4 `struct SPPMPixel::VisiblePoint`.
#[derive(Clone)]
struct VisiblePoint {
    p: Point3f,
    wo: Vector3f,
    bsdf: BSDF,
    beta: SampledSpectrum,
    secondary_lambda_terminated: bool,
}

/// pbrt-v4 `struct SPPMPixel`. Persistent across iterations.
struct SPPMPixel {
    /// Current search radius. Shrinks per iteration via the SPPM
    /// gamma=2/3 schedule: `r_new = r_old * sqrt((n + gamma*m) / (n + m))`.
    /// Mutex-wrapped so the update pass can write back the new radius.
    radius: Mutex<Float>,
    /// Direct light contribution accumulated across iterations (RGB).
    ld: Mutex<[Float; 3]>,
    /// Visible point set during the camera pass; consumed by the
    /// photon pass.
    vp: Mutex<Option<VisiblePoint>>,
    /// Sum of photon contributions, accumulated atomically across the
    phi_i: Mutex<[Float; 3]>,
    /// Photon count contributed this iteration. Mutex to allow parallel
    /// pixel updates without contention on a global atomic.
    m: Mutex<i32>,
    /// Running photon count (Hachisuka 2008 SPPM `N`).
    n: Mutex<Float>,
    /// Accumulated tau (post-radius-shrinking).
    tau: Mutex<[Float; 3]>,
}

impl SPPMPixel {
    fn new(radius: Float) -> Self {
        Self {
            radius: Mutex::new(radius),
            ld: Mutex::new([0.0; 3]),
            vp: Mutex::new(None),
            phi_i: Mutex::new([0.0; 3]),
            m: Mutex::new(0),
            n: Mutex::new(0.0),
            tau: Mutex::new([0.0; 3]),
        }
    }
}

impl SPPMIntegrator {
    pub fn new(
        scene: &Scene,
        camera: Arc<Camera>,
        sampler: Arc<RwLock<Sampler>>,
        initial_search_radius: Float,
        max_depth: i32,
        photons_per_iteration: i32,
        seed: i32,
    ) -> Self {
        let photons_per_iteration = if photons_per_iteration > 0 {
            photons_per_iteration
        } else {
            let film = camera.as_ref().get_film();
            let film = film.read().unwrap();
            film.pixel_bounds().area() as i32
        };
        Self {
            base: IntegratorBase::from_scene(scene),
            camera,
            sampler_prototype: sampler,
            initial_search_radius,
            max_depth,
            photons_per_iteration,
            seed,
        }
    }

    fn get_film(&self) -> Arc<RwLock<Film>> {
        self.camera.as_ref().get_film()
    }

    /// pbrt-v4 `SPPMIntegrator::SampleLd` (integrators.cpp:3279).
    /// Same shape as `PathIntegrator::SampleLd` -- BVH light sampling
    /// arm with MIS against bsdf.PDF, OffsetRayOrigin nudge.
    fn sample_ld(
        &self,
        intr: &SurfaceInteraction,
        bsdf: &BSDF,
        lambda: &SampledWavelengths,
        sampler: &mut Sampler,
        light_sampler: &LightSampler,
    ) -> SampledSpectrum {
        let mut ctx = LightSampleContext::from(&Interaction::from(intr));
        let flags = bsdf.flags();
        if is_reflective(flags) && !is_transmissive(flags) {
            ctx.p = offset_ray_origin(&intr.p, &intr.p_error, &intr.n, &intr.wo);
        } else if is_transmissive(flags) && !is_reflective(flags) {
            ctx.p = offset_ray_origin(&intr.p, &intr.p_error, &intr.n, &(-intr.wo));
        }
        let u = sampler.get_1d();
        let sampled_light = light_sampler.sample(&ctx, u);
        let u_light = sampler.get_2d();
        let sampled_light = match sampled_light {
            Some(s) => s,
            None => return SampledSpectrum::zero(),
        };
        let light = &sampled_light.light;
        let ls = light.as_ref().sample_li(&ctx, u_light, lambda, true);
        let ls = match ls {
            Some(s) => s,
            None => return SampledSpectrum::zero(),
        };
        if ls.l.is_black() || ls.pdf == 0.0 {
            return SampledSpectrum::zero();
        }
        let wo = intr.wo;
        let wi = ls.wi;
        let f = bsdf.f(wo, wi, TransportMode::Radiance)
            * Float::abs(Vector3f::dot(&wi, &Vector3f::from(intr.shading.n)));
        if f.is_black() || !self.base.unoccluded(&Interaction::from(intr), &ls.p_light) {
            return SampledSpectrum::zero();
        }
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

impl Integrator for SPPMIntegrator {
    fn render(&mut self) {
        let film = self.get_film();
        film.read().unwrap().render_start();

        let pixel_bounds = film.read().unwrap().pixel_bounds();
        let n_pixels = pixel_bounds.area() as usize;
        if n_pixels == 0 {
            let f = film.read().unwrap();
            f.render_end();
            f.write_image();
            return;
        }
        let width = (pixel_bounds.max.x - pixel_bounds.min.x) as usize;

        let pixels: Vec<SPPMPixel> = (0..n_pixels)
            .map(|_| SPPMPixel::new(self.initial_search_radius))
            .collect();

        let n_iterations = self.sampler_prototype.read().unwrap().samples_per_pixel() as i32;
        let samples_per_pixel = n_iterations as u32;
        let inv_sqrt_spp = 1.0 / (n_iterations as Float).sqrt();

        // Light samplers
        let bvh_light_sampler = LightSampler::BVH(BVHLightSampler::new(&self.base, 64));
        let shoot_light_sampler = PowerLightSampler::new(&self.base);

        // Halton digit permutations for photon shooting (pbrt-v4
        // `ComputeRadicalInversePermutations(seed)`, integrators.cpp:
        // SPPM::Render).
        let digit_perms = compute_radical_inverse_permutations(self.seed as u32);

        for iter in 0..n_iterations {
            // Wavelength + time samples for this pass
            let u_lambda = if PbrtOptions::get().disable_wavelength_jitter {
                0.5
            } else {
                radical_inverse(1, iter as u64)
            };
            let pass_lambda = film.read().unwrap().sample_wavelengths(u_lambda);
            let time_sample = radical_inverse(2, iter as u64);

            // ----- Visible-point pass (parallel over pixels) -----
            // pbrt-v4 (integrators.cpp:2872) wraps this in `ParallelFor2D`
            // over pixel tiles. We parallelize per-pixel via rayon with a
            // per-thread sampler clone + scratch buffer (rayon `for_each_init`).
            let sampler_proto = self.sampler_prototype.read().unwrap().clone();
            pixels.par_iter().enumerate().for_each_init(
                || (sampler_proto.clone(), MemoryArena::new()),
                |(sampler, scratch_buffer), (pi, pixel)| {
                    let px = pixel_bounds.min.x + (pi % width) as i32;
                    let py = pixel_bounds.min.y + (pi / width) as i32;
                    let p_pixel = Point2i::new(px, py);
                    sampler.start_pixel_sample(p_pixel, iter as u32, 0);
                    let mut cs = sampler.get_camera_sample(&p_pixel);
                    cs.time = time_sample;
                    let mut lambda = pass_lambda;
                    let crd = self.camera.generate_ray_differential(&cs, &lambda);
                    let Some(crd) = crd else { return };
                    if crd.weight == 0.0 {
                        return;
                    }
                    let mut beta = SampledSpectrum::new(crd.weight);
                    let mut ray = crd.ray;
                    ray.scale_differentials(inv_sqrt_spp);

                    let mut eta_scale: Float = 1.0;
                    let mut p_b: Float = 0.0;
                    let mut specular_bounce = true;
                    let mut have_set_vp = false;
                    let mut prev_intr_ctx = LightSampleContext::default();
                    let mut depth: i32 = 0;

                    loop {
                        let si = self.base.intersect(&ray.ray, Float::INFINITY);
                        // Accumulate contributions for ray with no intersection
                        let Some(mut si) = si else {
                            let mut l_pkt = SampledSpectrum::zero();
                            for light in self.base.infinite_lights.iter() {
                                let le = light.as_ref().le(&ray.ray, &lambda);
                                if le.is_black() {
                                    continue;
                                }
                                if depth == 0 || specular_bounce {
                                    l_pkt += beta * le;
                                } else {
                                    let p_l = bvh_light_sampler.pmf(&prev_intr_ctx, light)
                                        * light.as_ref().pdf_li(&prev_intr_ctx, ray.ray.d, true);
                                    let w_b = power_heuristic(1, p_b, 1, p_l);
                                    l_pkt += beta * w_b * le;
                                }
                            }
                            let rgb = film
                                .read()
                                .unwrap()
                                .base()
                                .pixel_sensor()
                                .to_output_rgb_from_packet(&l_pkt, &lambda);
                            let mut ld = pixel.ld.lock().unwrap();
                            ld[0] += rgb[0];
                            ld[1] += rgb[1];
                            ld[2] += rgb[2];
                            break;
                        };

                        // Get BSDF; skip medium boundaries
                        let bsdf = match si.intr.get_bsdf(
                            &ray,
                            self.camera.as_ref(),
                            sampler.samples_per_pixel(),
                            &mut lambda,
                            Some(sampler),
                        ) {
                            Some(b) => b,
                            None => {
                                // pbrt-v4 (integrators.cpp:2927-2930): skip
                                // medium boundaries while preserving ray
                                // differentials (for downstream texture
                                // filtering footprint).
                                si.intr.skip_intersection(&mut ray, si.t_hit);
                                continue;
                            }
                        };

                        // Surface emission with MIS
                        let mut l_pkt = SampledSpectrum::zero();
                        let le = si.intr.le(-ray.ray.d, &lambda);
                        if !le.is_black() {
                            if depth == 0 || specular_bounce {
                                l_pkt += beta * le;
                            } else if let Some(area_light) = si.intr.get_area_light() {
                                let p_l = bvh_light_sampler.pmf(&prev_intr_ctx, &area_light)
                                    * area_light.as_ref().pdf_li(&prev_intr_ctx, ray.ray.d, true);
                                let w_l = power_heuristic(1, p_b, 1, p_l);
                                l_pkt += beta * w_l * le;
                            }
                        }
                        if !l_pkt.is_black() {
                            let rgb = film
                                .read()
                                .unwrap()
                                .base()
                                .pixel_sensor()
                                .to_output_rgb_from_packet(&l_pkt, &lambda);
                            let mut ld = pixel.ld.lock().unwrap();
                            ld[0] += rgb[0];
                            ld[1] += rgb[1];
                            ld[2] += rgb[2];
                        }

                        // v4 `depth++ == maxDepth || haveSetVisiblePoint`.
                        let d_current = depth;
                        depth += 1;
                        if d_current == self.max_depth || have_set_vp {
                            break;
                        }

                        // Direct lighting at this hit (added to Ld)
                        let ld_pkt =
                            self.sample_ld(&si.intr, &bsdf, &lambda, sampler, &bvh_light_sampler);
                        if !ld_pkt.is_black() {
                            let contrib = beta * ld_pkt;
                            let rgb = film
                                .read()
                                .unwrap()
                                .base()
                                .pixel_sensor()
                                .to_output_rgb_from_packet(&contrib, &lambda);
                            let mut ld = pixel.ld.lock().unwrap();
                            ld[0] += rgb[0];
                            ld[1] += rgb[1];
                            ld[2] += rgb[2];
                        }

                        // Possibly create visible point and end camera path
                        let flags = bsdf.flags();
                        let wo = -ray.ray.d;
                        if is_diffuse(flags) || (is_glossy(flags) && depth == self.max_depth) {
                            let secondary_terminated = lambda.secondary_terminated();
                            *pixel.vp.lock().unwrap() = Some(VisiblePoint {
                                p: si.intr.p,
                                wo,
                                bsdf: bsdf.clone(),
                                beta,
                                secondary_lambda_terminated: secondary_terminated,
                            });
                            have_set_vp = true;
                        }

                        // Sample BSDF
                        let u = sampler.get_1d();
                        let bs = bsdf.sample_f(
                            wo,
                            u,
                            sampler.get_2d(),
                            TransportMode::Radiance,
                            BXDF_ALL,
                        );
                        let bs = match bs {
                            Some(s) => s,
                            None => break,
                        };
                        specular_bounce = bs.is_specular();
                        if bs.is_transmission() {
                            eta_scale *= bs.eta * bs.eta;
                        }
                        beta *= bs.f
                            * (Float::abs(Vector3f::dot(
                                &bs.wi,
                                &Vector3f::from(si.intr.shading.n),
                            )) / bs.pdf);
                        p_b = if bs.pdf_is_proportional {
                            bsdf.pdf(wo, bs.wi, TransportMode::Radiance, BXDF_REFL_TRANS_ALL)
                        } else {
                            bs.pdf
                        };

                        // RR
                        let rr_beta = beta * eta_scale;
                        if rr_beta.max_component_value() < 1.0 {
                            let q = Float::max(0.05, 1.0 - rr_beta.max_component_value());
                            if sampler.get_1d() < q {
                                break;
                            }
                            beta /= 1.0 - q;
                        }
                        ray = si
                            .intr
                            .spawn_ray_with_differentials(&ray, &bsdf, &bs.wi, bs.flags, bs.eta);
                        prev_intr_ctx = LightSampleContext::from(&Interaction::from(&si.intr));
                    }
                    scratch_buffer.reset();
                },
            );

            // ----- Build spatial hash grid -----
            let mut grid_bounds = Bounds3f::default();
            let mut max_radius: Float = 0.0;
            let mut any_vp = false;
            for pixel in pixels.iter() {
                let vp = pixel.vp.lock().unwrap();
                if let Some(vp) = vp.as_ref() {
                    if vp.beta.is_black() {
                        continue;
                    }
                    let r = *pixel.radius.lock().unwrap();
                    let vp_bound = Bounds3f::expand(&Bounds3f::new(&vp.p, &vp.p), r);
                    if any_vp {
                        grid_bounds = Bounds3f::union(&grid_bounds, &vp_bound);
                    } else {
                        grid_bounds = vp_bound;
                        any_vp = true;
                    }
                    max_radius = max_radius.max(r);
                }
            }
            if !any_vp {
                continue;
            }

            // Grid resolution per dimension
            let diag = grid_bounds.max - grid_bounds.min;
            let max_diag = diag.x.max(diag.y).max(diag.z);
            let base_grid_res = if max_radius > 0.0 {
                (max_diag / max_radius).max(1.0) as i32
            } else {
                1
            };
            let grid_res = [
                ((base_grid_res as Float * diag.x / max_diag) as i32).max(1),
                ((base_grid_res as Float * diag.y / max_diag) as i32).max(1),
                ((base_grid_res as Float * diag.z / max_diag) as i32).max(1),
            ];

            let hash_size = next_prime(n_pixels);
            let grid: Vec<Mutex<Vec<usize>>> =
                (0..hash_size).map(|_| Mutex::new(Vec::new())).collect();

            // Insert visible points into grid cells (parallel over pixels).
            // pbrt-v4 (integrators.cpp:3022) parallelizes this; per-cell
            // mutex on `grid[h]` is the contention point.
            pixels.par_iter().enumerate().for_each(|(pi, pixel)| {
                let vp_lock = pixel.vp.lock().unwrap();
                let Some(vp) = vp_lock.as_ref() else { return };
                if vp.beta.is_black() {
                    return;
                }
                let r = *pixel.radius.lock().unwrap();
                let (p_min, _) = to_grid(vp.p - Vector3f::new(r, r, r), &grid_bounds, &grid_res);
                let (p_max, _) = to_grid(vp.p + Vector3f::new(r, r, r), &grid_bounds, &grid_res);
                for z in p_min.z..=p_max.z {
                    for y in p_min.y..=p_max.y {
                        for x in p_min.x..=p_max.x {
                            let h = (hash_point3i(Point3i::new(x, y, z)) as usize) % hash_size;
                            grid[h].lock().unwrap().push(pi);
                        }
                    }
                }
            });

            // ----- Photon pass (parallel over photon index) -----
            // pbrt-v4 (integrators.cpp:3062) parallelizes this; each photon's
            // sampler dim allocation is per-photon (Halton index = iter * N +
            // photon_index) so threads are independent. Scratch buffer is
            // per-thread via `for_each_init` and reset after each photon.
            (0..self.photons_per_iteration as u64)
                .into_par_iter()
                .for_each_init(
                    || MemoryArena::new(),
                    |scratch_photon, photon_index| {
                        let halton_index =
                            (iter as u64) * (self.photons_per_iteration as u64) + photon_index;
                        let mut halton_dim: u32 = 0;
                        // Inlined `Sample1D` (pbrt-v4 integrators.cpp:3074-3079).
                        macro_rules! hsample_1d {
                            () => {{
                                let perm = permutation_for_dimension(halton_dim, &digit_perms);
                                let u = scrambled_radical_inverse(halton_dim, halton_index, perm);
                                halton_dim += 1;
                                u
                            }};
                        }
                        macro_rules! hsample_2d {
                            () => {{
                                let perm0 = permutation_for_dimension(halton_dim, &digit_perms);
                                let perm1 = permutation_for_dimension(halton_dim + 1, &digit_perms);
                                let u = Point2f::new(
                                    scrambled_radical_inverse(halton_dim, halton_index, perm0),
                                    scrambled_radical_inverse(halton_dim + 1, halton_index, perm1),
                                );
                                halton_dim += 2;
                                u
                            }};
                        }
                        let ul = hsample_1d!();
                        let sampled_light = shoot_light_sampler.sample(ul);
                        let Some(sl) = sampled_light else { return };
                        let light = sl.light.clone();
                        let p_l = sl.p;
                        let u_light0 = hsample_2d!();
                        let u_light1 = hsample_2d!();
                        let (sh_open, sh_close) = self.camera.get_shutter();
                        let u_light_time = lerp(time_sample, sh_open, sh_close);

                        let mut lambda = pass_lambda;
                        let les =
                            light
                                .as_ref()
                                .sample_le(u_light0, u_light1, &lambda, u_light_time);
                        let Some(les) = les else { return };
                        if les.pdf_pos == 0.0 || les.pdf_dir == 0.0 || les.l.is_black() {
                            return;
                        }
                        let mut photon_ray = RayDifferential::from(les.ray.clone());
                        let mut beta = les.l * les.abs_cos_theta(photon_ray.ray.d)
                            / (p_l * les.pdf_pos * les.pdf_dir);
                        if beta.is_black() {
                            return;
                        }

                        // pbrt-v4 (integrators.cpp:3147) uses `for (int depth = 0;
                        // depth < maxDepth; ++depth)` and compensates the medium-skip
                        // path with `--depth`. Rust's for-range auto-increments on
                        // `continue`, so we use an explicit while loop and only
                        // increment when a real bounce was made.
                        let mut depth: i32 = 0;
                        while depth < self.max_depth {
                            let si = self.base.intersect(&photon_ray.ray, Float::INFINITY);
                            let Some(mut si) = si else { break };

                            if depth > 0 {
                                // Add photon contribution to nearby visible points
                                let (p_grid, in_bounds) =
                                    to_grid(si.intr.p, &grid_bounds, &grid_res);
                                if in_bounds {
                                    let h = (hash_point3i(p_grid) as usize) % hash_size;
                                    let cell = grid[h].lock().unwrap();
                                    for &pi in cell.iter() {
                                        let pixel = &pixels[pi];
                                        let vp_lock = pixel.vp.lock().unwrap();
                                        let Some(vp) = vp_lock.as_ref() else { continue };
                                        let r = *pixel.radius.lock().unwrap();
                                        if Vector3f::distance_squared(&vp.p, &si.intr.p) > r * r {
                                            continue;
                                        }
                                        // pbrt-v4 (integrators.cpp:3142) evaluates the VP's
                                        // BSDF in Radiance mode (the default for `BSDF::f`).
                                        // The VP is anchored on the camera subpath, so the
                                        // photon's contribution is `Lo = f * Li * cos`, i.e.
                                        // a radiance evaluation, not an importance one.
                                        let wi = -photon_ray.ray.d;
                                        let phi =
                                            beta * vp.bsdf.f(vp.wo, wi, TransportMode::Radiance);
                                        let mut photon_lambda = lambda;
                                        if vp.secondary_lambda_terminated {
                                            photon_lambda.terminate_secondary();
                                        }
                                        let contrib = vp.beta * phi;
                                        let rgb = film
                                            .read()
                                            .unwrap()
                                            .base()
                                            .pixel_sensor()
                                            .to_output_rgb_from_packet(&contrib, &photon_lambda);
                                        let mut phi_i = pixel.phi_i.lock().unwrap();
                                        phi_i[0] += rgb[0];
                                        phi_i[1] += rgb[1];
                                        phi_i[2] += rgb[2];
                                        let mut m = pixel.m.lock().unwrap();
                                        *m += 1;
                                    }
                                }
                            }

                            let photon_bsdf = match si.intr.get_bsdf(
                                &photon_ray,
                                self.camera.as_ref(),
                                samples_per_pixel,
                                &mut lambda,
                                None,
                            ) {
                                Some(b) => b,
                                None => {
                                    si.intr.skip_intersection(&mut photon_ray, si.t_hit);
                                    continue;
                                }
                            };

                            let wo = -photon_ray.ray.d;
                            let uc = hsample_1d!();
                            let u2 = hsample_2d!();
                            let bs = photon_bsdf.sample_f(
                                wo,
                                uc,
                                u2,
                                TransportMode::Importance,
                                BXDF_ALL,
                            );
                            let Some(bs) = bs else { break };
                            let bnew = beta
                                * bs.f
                                * (Float::abs(Vector3f::dot(
                                    &bs.wi,
                                    &Vector3f::from(si.intr.shading.n),
                                )) / bs.pdf);
                            let beta_ratio = if beta.max_component_value() > 0.0 {
                                bnew.max_component_value() / beta.max_component_value()
                            } else {
                                0.0
                            };
                            let q = Float::max(0.0, 1.0 - beta_ratio);
                            if hsample_1d!() < q {
                                break;
                            }
                            beta = bnew / (1.0 - q);
                            photon_ray = si.intr.spawn_ray(&bs.wi).into();
                            depth += 1;
                        }
                        scratch_photon.reset();
                    },
                );

            // ----- Update pixel values from this pass's photons -----
            // pbrt-v4 SPPM update pass (integrators.cpp:3195-3217).
            // Per pixel: if m > 0, advance (n, tau, radius) via the
            // gamma=2/3 schedule and reset per-iteration accumulators;
            // then reset the visible point.
            let gamma: Float = 2.0 / 3.0;
            pixels.par_iter().for_each(|pixel| {
                let m = {
                    let mut m = pixel.m.lock().unwrap();
                    let v = *m;
                    *m = 0;
                    v
                };
                if m > 0 {
                    let phi_i = {
                        let mut phi = pixel.phi_i.lock().unwrap();
                        let v = *phi;
                        *phi = [0.0; 3];
                        v
                    };
                    let mut n = pixel.n.lock().unwrap();
                    let mut r = pixel.radius.lock().unwrap();
                    let n_new = *n + gamma * m as Float;
                    let r_new = *r * Float::sqrt(n_new / (*n + m as Float));
                    let mut tau = pixel.tau.lock().unwrap();
                    let scale = (r_new * r_new) / (*r * *r);
                    tau[0] = (tau[0] + phi_i[0]) * scale;
                    tau[1] = (tau[1] + phi_i[1]) * scale;
                    tau[2] = (tau[2] + phi_i[2]) * scale;
                    *n = n_new;
                    *r = r_new;
                }
                *pixel.vp.lock().unwrap() = None;
            });

            // ----- Per-iteration preview flush -----
            // Reset the film's pixel accumulator, rebuild a fresh
            // partial image from the current ld / tau / radius state,
            // and push it to the registered displays. The final
            // iteration's flush leaves the film holding the final
            // image, so no post-loop accumulation is needed.
            let iters_so_far = iter + 1;
            let n_photons_so_far = (iters_so_far as u64) * (self.photons_per_iteration as u64);
            let inv_iters = 1.0 / iters_so_far as Float;
            {
                let mut f = film.write().unwrap();
                f.clear();
            }
            {
                let f = film.read().unwrap();
                for py in pixel_bounds.min.y..pixel_bounds.max.y {
                    for px in pixel_bounds.min.x..pixel_bounds.max.x {
                        let p_pixel = Point2i::new(px, py);
                        let pi = (py - pixel_bounds.min.y) as usize * width
                            + (px - pixel_bounds.min.x) as usize;
                        let pixel = &pixels[pi];
                        let ld = *pixel.ld.lock().unwrap();
                        let tau = *pixel.tau.lock().unwrap();
                        let r = *pixel.radius.lock().unwrap();
                        let denom = (n_photons_so_far as Float) * PI * r * r;
                        let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                        let rgb = [
                            ld[0] * inv_iters + tau[0] * inv_denom,
                            ld[1] * inv_iters + tau[1] * inv_denom,
                            ld[2] * inv_iters + tau[2] * inv_denom,
                        ];
                        f.add_pixel_rgb(p_pixel, rgb, 1.0);
                    }
                }
                f.update_display(&pixel_bounds);
            }
        }

        let f = film.read().unwrap();
        f.render_end();
        f.write_image();
    }

    fn get_camera(&self) -> Arc<Camera> {
        self.camera.clone()
    }
}

unsafe impl Sync for SPPMIntegrator {}

/// pbrt-v4 SPPM `ToGrid(p, gridBounds, gridRes, &pi)`.
fn to_grid(p: Point3f, bounds: &Bounds3f, grid_res: &[i32; 3]) -> (Point3i, bool) {
    let pg = bounds.offset(&p);
    let mut pi = Point3i::new(0, 0, 0);
    let mut in_bounds = true;
    for i in 0..3 {
        let coord = (grid_res[i] as Float * pg[i]) as i32;
        let clamped = coord.clamp(0, grid_res[i] - 1);
        if coord < 0 || coord >= grid_res[i] {
            in_bounds = false;
        }
        match i {
            0 => pi.x = clamped,
            1 => pi.y = clamped,
            _ => pi.z = clamped,
        }
    }
    (pi, in_bounds)
}

/// Trivial multiplicative hash for Point3i. pbrt-v4 uses `Hash(...)` on a
/// 12-byte buffer; r4 keeps a simple wrap-safe mix here.
fn hash_point3i(p: Point3i) -> u64 {
    let x = p.x as u32 as u64;
    let y = p.y as u32 as u64;
    let z = p.z as u32 as u64;
    let mut h = x.wrapping_mul(73856093);
    h ^= y.wrapping_mul(19349663);
    h ^= z.wrapping_mul(83492791);
    h.wrapping_mul(0x100000001b3)
}

/// pbrt-v4 `NextPrime(n)` -- returns the next prime >= n.
fn next_prime(mut n: usize) -> usize {
    if n <= 2 {
        return 2;
    }
    if n % 2 == 0 {
        n += 1;
    }
    while !is_prime(n) {
        n += 2;
    }
    n
}

fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let mut i: usize = 3;
    while i.saturating_mul(i) <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

pub fn create_sppm_integrator(
    params: &ParameterDictionary,
    sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    let max_depth = params.get_one_int("maxdepth", 5);
    let photons_per_iteration = params.get_one_int("photonsperiteration", -1);
    let initial_search_radius = params.get_one_float("radius", 1.0);
    let seed = params.get_one_int("seed", 0);
    Ok(Arc::new(RwLock::new(SPPMIntegrator::new(
        scene,
        camera.clone(),
        sampler.clone(),
        initial_search_radius,
        max_depth,
        photons_per_iteration,
        seed,
    ))))
}
