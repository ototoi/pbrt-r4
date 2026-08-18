//! pbrt-v4 `class Integrator` is the base of the Integrator hierarchy.
//! In v4 it owns the scene data (`aggregate`, `lights`,
//! `infiniteLights`) directly; intersection queries (`Intersect`,
//! `IntersectP`, `IntersectTr`) live on the base too. r4 mirrors this
//! with `IntegratorBase`, which every Integrator embeds via the
//! `ImageTileIntegratorBase -> RayIntegratorBase` chain.
//!
//! Construction note: r4's `Scene` is built first (parser side) and
//! `IntegratorBase` Arc-clones its `aggregate` / `lights` /
//! `infinite_lights` into the integrator. After that the integrator
//! is self-sufficient -- it doesn't need to be handed a `&Scene` at
//! render time for traversal queries.

use crate::base::camera::Camera;
use crate::base::light::Light;
use crate::base::medium::sample_t_maj;
use crate::cpu::primitive::*;
use crate::interaction::{Interaction, ShapeIntersection};
use crate::scene::*;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::rng::RNG;
use crate::util::spectrum::*;

use std::sync::Arc;

/// Equivalent of pbrt-v4 `class Integrator`'s data members: the scene
/// geometry aggregate and the light lists. Held by Arc so each
/// integrator can keep its own clones without copying the BVH or
/// duplicating light vectors.
pub struct IntegratorBase {
    pub aggregate: Arc<Primitive>,
    pub lights: Vec<Arc<Light>>,
    pub infinite_lights: Vec<Arc<Light>>,
}

impl IntegratorBase {
    /// Arc-clone scene data into the integrator. After this returns
    /// the integrator no longer depends on the `Scene` value.
    pub fn from_scene(scene: &Scene) -> Self {
        Self {
            aggregate: Arc::clone(&scene.aggregate),
            lights: scene.lights.iter().cloned().collect(),
            infinite_lights: scene.infinite_lights.iter().cloned().collect(),
        }
    }

    pub fn world_bound(&self) -> Bounds3f {
        self.aggregate.bounds()
    }

    /// Direct ray intersection -- pbrt-v4 `Integrator::Intersect`.
    pub fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        self.aggregate.intersect(ray, t_max)
    }

    /// Shadow ray test -- pbrt-v4 `Integrator::IntersectP`.
    pub fn intersect_p(&self, ray: &Ray, t_max: Float) -> bool {
        self.aggregate.intersect_p(ray, t_max)
    }

    /// pbrt-v4 `bool Integrator::Unoccluded(const Interaction &p0, const
    /// Interaction &p1) const` (integrators.h:52). Shadow-ray test along
    /// the segment p0 -> p1 (clipped just before the endpoint).
    pub fn unoccluded(&self, p0: &Interaction, p1: &Interaction) -> bool {
        !self.intersect_p(&p0.spawn_ray_to(p1), 1.0 - SHADOW_EPSILON)
    }

    pub fn tr(
        &self,
        p0: &Interaction,
        p1: &Interaction,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        let mut rng = RNG::new();
        rng.set_sequence_with_seed(hash_point3f(&p0.get_p()), hash_point3f(&p1.get_p()));

        let mut ray = p0.spawn_ray_to(p1);
        let mut tr = SampledSpectrum::one();
        let mut inv_w = SampledSpectrum::one();
        if ray.d.length_squared() == 0.0 {
            return SampledSpectrum::one();
        }

        loop {
            let si = self.intersect(&ray, 1.0 - SHADOW_EPSILON);
            if let Some(si) = si.as_ref() {
                if si.intr.get_material().is_some() {
                    return SampledSpectrum::zero();
                }
            }

            if let Some(medium) = ray.medium.as_ref() {
                let p_exit = ray.position(
                    si.as_ref()
                        .map(|si| si.t_hit)
                        .unwrap_or(1.0 - SHADOW_EPSILON),
                );
                ray.d = p_exit - ray.o;
                let t_maj_final = sample_t_maj(
                    medium.as_ref(),
                    &ray,
                    1.0,
                    rng.uniform_float(),
                    lambda,
                    &mut rng,
                    |_p, mp, sigma_maj, t_maj, _rng| {
                        let sigma_n = clamp_zero(sigma_maj - mp.sigma_a - mp.sigma_s);
                        let pr = t_maj[0] * sigma_maj[0];
                        if pr <= 0.0 {
                            tr = SampledSpectrum::zero();
                            inv_w = SampledSpectrum::zero();
                            return false;
                        }
                        tr *= t_maj * sigma_n / pr;
                        inv_w *= t_maj * sigma_maj / pr;
                        tr.average() != 0.0 && inv_w.average() != 0.0
                    },
                );
                if t_maj_final[0] != 0.0 {
                    tr *= t_maj_final / t_maj_final[0];
                    inv_w *= t_maj_final / t_maj_final[0];
                } else {
                    return SampledSpectrum::zero();
                }
            }

            if let Some(si) = si {
                ray = Interaction::from(&si.intr).spawn_ray_to(p1);
            } else {
                break;
            }
        }

        let inv_w_avg = inv_w.average();
        if inv_w_avg == 0.0 {
            SampledSpectrum::zero()
        } else {
            tr / inv_w_avg
        }
    }
}

unsafe impl Sync for IntegratorBase {}

pub trait Integrator {
    fn render(&mut self);
    fn get_camera(&self) -> Arc<Camera>;
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
