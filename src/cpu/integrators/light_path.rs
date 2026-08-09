use crate::base::bxdf::{TransportMode, BXDF_ALL};
use crate::base::camera::Camera;
use crate::base::lightsampler::{LightSampleContext, LightSampler};
use crate::base::sampler::Sampler;
use crate::cpu::integrators::*;
use crate::film::Film;
use crate::interaction::Interaction;
use crate::options::PbrtOptions;
use crate::paramdict::*;
use crate::scene::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::memory::*;
use crate::util::spectrum::*;

use std::sync::Arc;
use std::sync::RwLock;

pub struct LightPathIntegrator {
    base: ImageTileIntegratorBase,
    pixel_bounds: Bounds2i,
    max_depth: i32,
    light_sample_strategy: String,
    light_sampler: Option<LightSampler>,
    // pbrt-v4 `Allocator` is no-op in r4; `IntegratorBase` (held inside
    // `base`) supplies `intersect`/`unoccluded`.
}

impl LightPathIntegrator {
    pub fn new(
        max_depth: i32,
        scene: &Scene,
        camera: &Arc<Camera>,
        sampler: &Arc<RwLock<Sampler>>,
        pixel_bounds: &Bounds2i,
        light_sample_strategy: &str,
    ) -> Self {
        LightPathIntegrator {
            base: ImageTileIntegratorBase::new(scene, camera, sampler),
            pixel_bounds: *pixel_bounds,
            max_depth: max_depth.max(0),
            light_sample_strategy: light_sample_strategy.to_string(),
            light_sampler: None,
        }
    }

    /// pbrt-v4 `LightPathIntegrator::EvaluatePixelSample`
    /// (integrators.cpp:507-597). Line-by-line translation.
    fn trace_light_path(
        &self,
        sampler: &mut Sampler,
        _scratch_buffer: &mut MemoryArena,
        film: &Arc<RwLock<Film>>,
    ) {
        let light_sampler = match self.light_sampler.as_ref() {
            Some(ls) => ls,
            None => return,
        };
        let camera = self.base.camera.as_ref();

        // Sample wavelengths for the ray
        let lu = if PbrtOptions::get().disable_wavelength_jitter {
            0.5
        } else {
            sampler.get_1d()
        };
        let mut lambda = film.read().unwrap().sample_wavelengths(lu);

        // Sample light to start light path
        let ul = sampler.get_1d();
        let default_ctx = LightSampleContext::default();
        let sampled_light = match light_sampler.sample(&default_ctx, ul) {
            Some(sl) => sl,
            None => return,
        };
        let light = sampled_light.light.clone();
        let p_l = sampled_light.p;

        let (so, sc) = camera.get_shutter();
        let time = lerp(sampler.get_1d(), so, sc);
        let ul0 = sampler.get_2d();
        let ul1 = sampler.get_2d();
        let les = match light.as_ref().sample_le(ul0, ul1, &lambda, time) {
            Some(s) => s,
            None => return,
        };
        if les.pdf_pos == 0.0 || les.pdf_dir == 0.0 || les.l.is_black() {
            return;
        }

        // Add contribution of directly visible light source
        if let Some(intr) = les.intr.as_ref() {
            let cs = camera.sample_wi(intr, &sampler.get_2d(), &lambda);
            if let Some(cs) = cs {
                if cs.pdf != 0.0 {
                    // v4 `light.PDF_Li(cs->pLens, -cs->wi)`: the receiver
                    // context here is the camera lens; r4 doesn't expose a
                    // pLens, so we build a minimal LightSampleContext from
                    // the visibility tester's far-end interaction.
                    let lens_ctx = LightSampleContext::from(&cs.visibility.p1);
                    let pdf = light.as_ref().pdf_li(&lens_ctx, -cs.wi, false);
                    if pdf > 0.0 {
                        // Add light's emitted radiance if nonzero and light is visible
                        let uv = intr
                            .as_surface_interaction()
                            .map(|si| si.uv)
                            .unwrap_or(Point2f::new(0.0, 0.0));
                        let le = light
                            .as_ref()
                            .l(intr.get_p(), intr.get_n(), uv, cs.wi, &lambda);
                        if !le.is_black() && cs.visibility.unoccluded(&self.base.base) {
                            // Compute visible light's path contribution and add to film
                            let dist_sq = Vector3f::distance_squared(
                                &intr.get_p(),
                                &cs.visibility.p1.get_p(),
                            );
                            let wi_spec = cs.wi_spec.sample(&lambda);
                            let l = le * wi_spec * (dist_sq / (p_l * pdf * cs.pdf));
                            film.write().unwrap().add_splat_packet(
                                &Vector2f::new(cs.p_raster.x, cs.p_raster.y),
                                &l,
                                &lambda,
                            );
                        }
                    }
                }
            }
        }

        // Follow light path and accumulate contributions to image
        let mut depth: i32 = 0;
        // Initialize light path ray and weighted path throughput `beta`
        let mut ray = RayDifferential::from(les.ray.clone());
        let mut beta = les.l * les.abs_cos_theta(ray.ray.d) / (p_l * les.pdf_pos * les.pdf_dir);

        loop {
            // Intersect light path ray with scene
            let si = match self.base.base.intersect(&ray.ray, Float::INFINITY) {
                Some(s) => s,
                None => break,
            };
            let mut isect = si.intr;

            // Get BSDF and skip over medium boundaries
            let bsdf = match isect.get_bsdf(
                &ray,
                camera,
                sampler.samples_per_pixel(),
                &mut lambda,
                Some(sampler),
            ) {
                Some(b) => b,
                None => {
                    ray = isect.spawn_ray(&ray.ray.d).into();
                    continue;
                }
            };

            // End path if maximum depth reached
            let d_current = depth;
            depth += 1;
            if d_current == self.max_depth {
                break;
            }

            // Splat contribution into film if intersection point is visible to camera
            let u = sampler.get_2d();
            let cs = camera.sample_wi(&Interaction::from(&isect), &u, &lambda);
            if let Some(cs) = cs {
                if cs.pdf != 0.0 {
                    let f = bsdf.f(isect.wo, cs.wi, TransportMode::Importance);
                    let cos = Float::abs(Vector3f::dot(&cs.wi, &Vector3f::from(isect.shading.n)));
                    let wi_spec = cs.wi_spec.sample(&lambda);
                    let l = beta * f * cos * wi_spec / cs.pdf;
                    if !l.is_black() && cs.visibility.unoccluded(&self.base.base) {
                        film.write().unwrap().add_splat_packet(
                            &Vector2f::new(cs.p_raster.x, cs.p_raster.y),
                            &l,
                            &lambda,
                        );
                    }
                }
            }

            // Sample BSDF and update light path state
            let uc = sampler.get_1d();
            let bs = match bsdf.sample_f(
                isect.wo,
                uc,
                sampler.get_2d(),
                TransportMode::Importance,
                BXDF_ALL,
            ) {
                Some(s) => s,
                None => break,
            };
            beta *= bs.f
                * (Float::abs(Vector3f::dot(&bs.wi, &Vector3f::from(isect.shading.n))) / bs.pdf);
            ray = isect.spawn_ray_with_differentials(&ray, &bsdf, &bs.wi, bs.flags, bs.eta);
        }
    }
}

impl Integrator for LightPathIntegrator {
    fn render(&mut self) {
        match LightSampler::create(&self.light_sample_strategy, &self.base.base) {
            Ok(ls) => self.light_sampler = Some(ls),
            Err(e) => {
                log::warn!("LightPathIntegrator: {:?}", e);
                return;
            }
        }
        // pbrt-v4 ImageTileIntegrator::Render (integrators.cpp:66): tile
        // loop drives EvaluatePixelSample. LightPath's evaluate_pixel_sample
        // splats directly and returns None so the FilmTile update is
        // skipped per-sample. After the tile sweep, the base
        // `SampleIntegratorCore::render` flushes the splat tiles with
        // `1 / spp` scaling (matching pbrt-v4 `WriteImage(metadata,
        // 1/spp)`) and writes the image, so no extra work is needed here.
        let camera = self.base.camera.clone();
        let film = camera.get_film();
        let sampler = Arc::clone(&self.base.sampler);
        SampleIntegratorCore::render(self, camera.as_ref(), &film, &sampler);
    }

    fn get_camera(&self) -> Arc<Camera> {
        self.base.camera.clone()
    }
}

impl ImageTileIntegrator for LightPathIntegrator {
    fn evaluate_pixel_sample(
        &self,
        _p_pixel: Point2i,
        _sample_index: i32,
        sampler: &mut Sampler,
        arena: &mut MemoryArena,
    ) -> Option<PixelSample> {
        let film = self.base.camera.get_film();
        self.trace_light_path(sampler, arena, &film);
        // LightPath splats directly; nothing for the tile loop to add.
        let _ = self.pixel_bounds;
        None
    }
}

unsafe impl Sync for LightPathIntegrator {}

pub fn create_light_path_integrator(
    params: &ParameterDictionary,
    sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    let pixel_bounds = camera.get_film().read().unwrap().pixel_bounds();
    let max_depth = params.get_one_int("maxdepth", 5);
    let light_sample_strategy = params.get_one_string("lightsampler", "power");
    Ok(Arc::new(RwLock::new(LightPathIntegrator::new(
        max_depth,
        scene,
        camera,
        sampler,
        &pixel_bounds,
        &light_sample_strategy,
    ))))
}
