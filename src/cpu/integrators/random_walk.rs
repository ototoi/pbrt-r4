// pbrt-v4 verbatim translation of `class RandomWalkIntegrator`
// (integrators.h:114-180). The simplest path tracer in v4: no MIS, no
// explicit direct-lighting connection, no Russian roulette. At each
// surface hit we add Le and recurse along a uniform-sphere-sampled
// direction (1 / (4 pi) pdf).

use crate::base::bxdf::TransportMode;
use crate::base::camera::Camera;
use crate::base::sampler::Sampler;
use crate::cpu::integrators::*;
use crate::film::VisibleSurface;
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

pub struct RandomWalkIntegrator {
    base: RayIntegratorBase,
    max_depth: i32,
}

impl RandomWalkIntegrator {
    pub fn new(
        max_depth: i32,
        scene: &Scene,
        camera: &Arc<Camera>,
        sampler: &Arc<RwLock<Sampler>>,
        pixel_bounds: &Bounds2i,
    ) -> Self {
        RandomWalkIntegrator {
            base: RayIntegratorBase::new(scene, camera, sampler, pixel_bounds),
            max_depth: max_depth.max(0),
        }
    }

    /// pbrt-v4 `SampledSpectrum RandomWalkIntegrator::LiRandomWalk(
    /// RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
    /// ScratchBuffer &scratchBuffer, int depth) const`
    /// (integrators.h:136). Line-by-line translation.
    fn li_random_walk(
        &self,
        ray: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut MemoryArena,
        depth: i32,
    ) -> SampledSpectrum {
        // Intersect ray with scene and return if no intersection
        let Some(mut si) = self.base.intersect(&ray.ray, Float::INFINITY) else {
            // Return emitted light from infinite light sources
            let mut le = SampledSpectrum::zero();
            for light in self.base.infinite_lights.iter() {
                le += light.as_ref().le(&ray.ray, lambda);
            }
            return le;
        };

        // Get emitted radiance at surface intersection
        let wo = -ray.ray.d;
        let le = si.intr.le(wo, lambda);

        // Terminate random walk if maximum depth has been reached
        if depth == self.max_depth {
            return le;
        }

        // Compute BSDF at random walk intersection point
        let Some(bsdf) = si.intr.get_bsdf(
            ray,
            self.base.camera.as_ref(),
            sampler.samples_per_pixel(),
            lambda,
            Some(sampler),
        ) else {
            return le;
        };

        // Randomly sample direction leaving surface for random walk
        let u = sampler.get_2d();
        let wp = uniform_sample_sphere(&u);

        // Evaluate BSDF at surface for sampled direction
        let fcos = bsdf.f(wo, wp, TransportMode::Radiance)
            * Float::abs(Vector3f::dot(&wp, &Vector3f::from(si.intr.shading.n)));
        if fcos.is_black() {
            return le;
        }

        // Recursively trace ray to estimate incident radiance at surface
        let next: RayDifferential = si.intr.spawn_ray(&wp).into();
        le + fcos * self.li_random_walk(&next, lambda, sampler, scratch_buffer, depth + 1)
            / (1.0 / (4.0 * PI))
    }
}

impl Integrator for RandomWalkIntegrator {
    fn render(&mut self) {
        RayIntegratorBase::render(self);
    }

    fn get_camera(&self) -> Arc<Camera> {
        self.base.camera.clone()
    }
}

impl RayIntegrator for RandomWalkIntegrator {
    /// pbrt-v4 inline `RandomWalkIntegrator::Li` (integrators.h:128):
    /// just calls `LiRandomWalk(..., 0)`.
    fn li(
        &self,
        ray: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut MemoryArena,
        _visible_surface: Option<&mut VisibleSurface>,
    ) -> SampledSpectrum {
        self.li_random_walk(ray, lambda, sampler, scratch_buffer, 0)
    }

    fn get_sampler(&self) -> Arc<RwLock<Sampler>> {
        Arc::clone(&self.base.sampler)
    }

    fn get_pixel_bounds(&self) -> Bounds2i {
        self.base.pixel_bounds
    }
}

crate::impl_image_tile_integrator_via_ray!(RandomWalkIntegrator);

unsafe impl Sync for RandomWalkIntegrator {}

pub fn create_random_walk_integrator(
    params: &ParameterDictionary,
    sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    let pixel_bounds = camera.get_film().read().unwrap().pixel_bounds();
    let max_depth = params.get_one_int("maxdepth", 5);
    Ok(Arc::new(RwLock::new(RandomWalkIntegrator::new(
        max_depth,
        scene,
        camera,
        sampler,
        &pixel_bounds,
    ))))
}
