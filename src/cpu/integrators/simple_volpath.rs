use crate::base::bxdf::{TransportMode, BXDF_ALL};
use crate::base::camera::Camera;
use crate::base::light::is_delta_light;
use crate::base::medium::{sample_t_maj_coefficients, MediumCoefficients};
use crate::base::sampler::Sampler;
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
use crate::util::spectrum::*;

use std::sync::Arc;
use std::sync::RwLock;

pub struct SimpleVolPathIntegrator {
    base: RayIntegratorBase,
    max_depth: i32,
}

impl SimpleVolPathIntegrator {
    pub fn new(
        max_depth: i32,
        scene: &Scene,
        camera: &Arc<Camera>,
        sampler: &Arc<RwLock<Sampler>>,
        pixel_bounds: &Bounds2i,
    ) -> Self {
        SimpleVolPathIntegrator {
            base: RayIntegratorBase::new(scene, camera, sampler, pixel_bounds),
            max_depth: max_depth.max(0),
        }
    }
}

impl Integrator for SimpleVolPathIntegrator {
    fn render(&mut self) {
        RayIntegratorBase::render(self);
    }
    fn get_camera(&self) -> Arc<Camera> {
        self.base.camera.clone()
    }
}

impl RayIntegrator for SimpleVolPathIntegrator {
    /// pbrt-v4 `SampledSpectrum SimpleVolPathIntegrator::Li(...)`
    /// (integrators.cpp:834-935). Line-by-line translation.
    fn li(
        &self,
        r: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        _scratch_buffer: &mut MemoryArena,
        _visible_surface: Option<&mut VisibleSurface>,
    ) -> SampledSpectrum {
        // Declare local variables for delta tracking integration
        let mut l = SampledSpectrum::zero();
        let mut beta: Float = 1.0;
        let mut depth: i32 = 0;
        let mut ray = r.clone();

        // Terminate secondary wavelengths before starting random walk
        lambda.terminate_secondary();

        loop {
            // Estimate radiance for ray path using delta tracking
            let si = self.base.intersect(&ray.ray, Float::INFINITY);
            let mut scattered = false;
            let mut terminated = false;
            if let Some(medium) = ray.ray.medium.as_ref() {
                // Initialize RNG for sampling the majorant transmittance
                let hash0 = hash_f32(sampler.get_1d());
                let hash1 = hash_f32(sampler.get_1d());
                let mut rng = RNG::new();
                rng.set_sequence_with_seed(hash0, hash1);

                // Sample medium using delta tracking
                let t_max = match si.as_ref() {
                    Some(s) => s.t_hit,
                    None => Float::INFINITY,
                };
                let u = sampler.get_1d();
                let mut u_mode = sampler.get_1d();

                // Capture mutable state for the closure via cells.
                use std::cell::Cell;
                let beta_cell = Cell::new(beta);
                let l_cell = Cell::new(l);
                let scattered_cell = Cell::new(false);
                let terminated_cell = Cell::new(false);
                let depth_cell = Cell::new(depth);
                let new_ray_o = Cell::new(ray.ray.o);
                let new_ray_d = Cell::new(ray.ray.d);

                let _ = sample_t_maj_coefficients(
                    medium.as_ref(),
                    &ray.ray,
                    t_max,
                    u,
                    &*lambda,
                    &mut rng,
                    |p, mp: MediumCoefficients, sigma_maj, _t_maj, rng| {
                        // Compute medium event probabilities (v4 lines 862-865).
                        let sa = mp.sigma_a;
                        let ss = mp.sigma_s;
                        let sigma_maj_ch = sigma_maj[0];
                        let p_absorb = if sigma_maj_ch > 0.0 {
                            sa[0] / sigma_maj_ch
                        } else {
                            0.0
                        };
                        let p_scatter = if sigma_maj_ch > 0.0 {
                            ss[0] / sigma_maj_ch
                        } else {
                            0.0
                        };
                        let p_null = Float::max(0.0, 1.0 - p_absorb - p_scatter);

                        // Randomly sample medium scattering event for delta tracking
                        let mode = sample_discrete3(&[p_absorb, p_scatter, p_null], u_mode);
                        if mode == 0 {
                            // Handle absorption event for medium sample
                            let le_packet = mp.le;
                            l_cell.set(l_cell.get() + le_packet * beta_cell.get());
                            terminated_cell.set(true);
                            return false;
                        } else if mode == 1 {
                            // Handle regular scattering event for medium sample.
                            // Stop path sampling if maximum depth has been reached.
                            let d_current = depth_cell.get();
                            depth_cell.set(d_current + 1);
                            if d_current >= self.max_depth {
                                terminated_cell.set(true);
                                return false;
                            }
                            // Sample phase function for medium scattering event.
                            let phase = medium.sample_phase_function(&p, lambda);
                            let u2 = Point2f::new(rng.uniform_float(), rng.uniform_float());
                            let (pdf, wi) = phase.sample_p(&-new_ray_d.get(), &u2);
                            if pdf == 0.0 {
                                terminated_cell.set(true);
                                return false;
                            }
                            // v4 `beta *= ps->p / ps->pdf` -- for HG, p == pdf so
                            // this is 1; for general phase functions r4 recovers
                            // `p` by re-evaluating `phase.p(wo, wi)`.
                            let p_val = phase.p(&-new_ray_d.get(), &wi);
                            beta_cell.set(beta_cell.get() * (p_val / pdf));
                            new_ray_o.set(p);
                            new_ray_d.set(wi);
                            scattered_cell.set(true);
                            return false;
                        } else {
                            // Handle null-scattering event for medium sample
                            u_mode = rng.uniform_float();
                            return true;
                        }
                    },
                );

                beta = beta_cell.get();
                l = l_cell.get();
                scattered = scattered_cell.get();
                terminated = terminated_cell.get();
                depth = depth_cell.get();
                if scattered {
                    ray.ray.o = new_ray_o.get();
                    ray.ray.d = new_ray_d.get();
                }
            }

            // Handle terminated and unscattered rays after medium sampling
            if terminated {
                return l;
            }
            if scattered {
                continue;
            }

            // Add emission to surviving ray
            let si = match si {
                Some(s) => s,
                None => {
                    for light in self.base.infinite_lights.iter() {
                        l += light.as_ref().le(&ray.ray, lambda) * beta;
                    }
                    return l;
                }
            };
            l += si.intr.le(-ray.ray.d, lambda) * beta;

            // Handle surface intersection along ray path
            let mut isect = si.intr;
            match isect.get_bsdf(
                &ray,
                self.base.camera.as_ref(),
                sampler.samples_per_pixel(),
                lambda,
                Some(sampler),
            ) {
                Some(bsdf) => {
                    let uc = sampler.get_1d();
                    let u = sampler.get_2d();
                    if bsdf
                        .sample_f(isect.wo, uc, u, TransportMode::Radiance, BXDF_ALL)
                        .is_some()
                    {
                        log::error!(
                            "SimpleVolPathIntegrator: surface scattering encountered; \
                             use VolPathIntegrator for scenes with surface BSDFs."
                        );
                        return SampledSpectrum::zero();
                    }
                    break;
                }
                None => {
                    // Medium boundary: advance through.
                    ray = isect.spawn_ray(&ray.ray.d).into();
                    continue;
                }
            };
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

crate::impl_image_tile_integrator_via_ray!(SimpleVolPathIntegrator);

unsafe impl Sync for SimpleVolPathIntegrator {}

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

fn hash_f32(x: Float) -> u64 {
    let bits = x.to_bits() as u64;
    // Splittable-64 style mix; matches Hash() output size used by v4.
    let mut z = bits.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

pub fn create_simple_volpath_integrator(
    params: &ParameterDictionary,
    sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    if scene
        .lights
        .iter()
        .any(|light| is_delta_light(light.light_type()))
    {
        return Err(PbrtError::error(
            "SimpleVolPathIntegrator only supports area and infinite light sources",
        ));
    }
    let pixel_bounds = camera.get_film().read().unwrap().pixel_bounds();
    let max_depth = params.get_one_int("maxdepth", 5);
    Ok(Arc::new(RwLock::new(SimpleVolPathIntegrator::new(
        max_depth,
        scene,
        camera,
        sampler,
        &pixel_bounds,
    ))))
}
