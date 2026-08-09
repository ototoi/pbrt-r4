use crate::base::shape::{Shape, ShapeSampleContext};
use crate::interaction::*;

use crate::textures::*;
use crate::util::base::*;
use crate::util::geometry::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

use std::sync::Arc;

#[derive(Clone)]
pub enum AlphaMaskInfo {
    Texture { texture: Arc<FloatTexture> },
    Value { value: Float },
}

enum AlphaMaskSource {
    Texture(Arc<FloatTexture>),
    Value(Float),
}

pub struct AlphaMaskShape {
    shape: Arc<Shape>,
    single_intersection: bool,
    test_intersection: bool,
    test_intersection_p: bool,
    constant_zero_alpha: bool,
    alpha_mask: Option<AlphaMaskSource>,
    shadow_alpha_mask: Option<AlphaMaskSource>,
}

impl AlphaMaskShape {
    pub fn new(
        shape: &Arc<Shape>,
        alpha_mask_info: &Option<AlphaMaskInfo>,
        shadow_alpha_mask_info: &Option<AlphaMaskInfo>,
    ) -> Self {
        let mut test_intersection = true;
        let mut test_intersection_p = true;
        let mut constant_zero_alpha = false;
        let mut alpha_mask = None;
        let mut shadow_alpha_mask = None;
        if let Some(info) = alpha_mask_info.as_ref() {
            match info {
                AlphaMaskInfo::Texture { texture } => {
                    alpha_mask = Some(AlphaMaskSource::Texture(Arc::clone(texture)));
                }
                AlphaMaskInfo::Value { value: alpha } => {
                    alpha_mask = Some(AlphaMaskSource::Value(*alpha));
                    if *alpha <= 0.0 {
                        test_intersection = false;
                        test_intersection_p = false;
                        constant_zero_alpha = *alpha == 0.0;
                    }
                }
            }
        }
        if let Some(info) = shadow_alpha_mask_info.as_ref() {
            match info {
                AlphaMaskInfo::Texture { texture } => {
                    shadow_alpha_mask = Some(AlphaMaskSource::Texture(Arc::clone(texture)));
                }
                AlphaMaskInfo::Value { value: alpha } => {
                    shadow_alpha_mask = Some(AlphaMaskSource::Value(*alpha));
                    if *alpha <= 0.0 {
                        test_intersection_p = false;
                    }
                }
            }
        }
        AlphaMaskShape {
            shape: Arc::clone(shape),
            single_intersection: matches!(shape.as_ref(), Shape::Triangle(_)),
            test_intersection,
            test_intersection_p,
            constant_zero_alpha,
            alpha_mask,
            shadow_alpha_mask,
        }
    }
}

impl AlphaMaskShape {
    pub fn object_bound(&self) -> Bounds3f {
        let shape = self.shape.as_ref();
        return shape.object_bound();
    }
    pub fn world_bound(&self) -> Bounds3f {
        let shape = self.shape.as_ref();
        return shape.world_bound();
    }

    /// AlphaMask is a wrapper -- delegate to the inner shape.
    pub fn normal_bounds(&self) -> DirectionCone {
        self.shape.as_ref().normal_bounds()
    }

    pub fn has_constant_zero_alpha(&self) -> bool {
        self.constant_zero_alpha
    }

    pub fn alpha(&self, inter: &Interaction) -> Option<Float> {
        let si = inter.as_surface_interaction()?;
        let ctx = TextureEvalContext::from(si);
        self.alpha_mask
            .as_ref()
            .map(|source| match source {
                AlphaMaskSource::Texture(texture) => texture.evaluate(&ctx),
                AlphaMaskSource::Value(value) => *value,
            })
            .or_else(|| self.constant_zero_alpha.then_some(0.0))
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        if !self.test_intersection {
            return None;
        }

        let mut ray = r.clone();
        let mut t_offset = 0.0;
        loop {
            let isect = self.shape.intersect(&ray, t_max - t_offset)?;
            if self.accept_alpha(&isect.intr, &ray) {
                return Some(ShapeIntersection::new(isect.intr, isect.t_hit + t_offset));
            }

            if self.single_intersection {
                return None;
            }

            if isect.t_hit <= 0.0 {
                return None;
            }
            t_offset += isect.t_hit;
            if t_offset >= t_max {
                return None;
            }
            ray = isect.intr.spawn_ray(&ray.d);
        }
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        if !self.test_intersection_p {
            return false;
        }

        let mut ray = r.clone();
        let mut t_offset = 0.0;
        loop {
            let Some(isect) = self.shape.intersect(&ray, t_max - t_offset) else {
                return false;
            };
            if self.accept_alpha(&isect.intr, &ray) && self.accept_shadow_alpha(&isect.intr, &ray) {
                return true;
            }

            if self.single_intersection {
                return false;
            }

            if isect.t_hit <= 0.0 {
                return false;
            }
            t_offset += isect.t_hit;
            if t_offset >= t_max {
                return false;
            }
            ray = isect.intr.spawn_ray(&ray.d);
        }
    }

    fn accept_alpha(&self, intr: &SurfaceInteraction, ray: &Ray) -> bool {
        self.accept_mask(self.alpha_mask.as_ref(), intr, ray)
    }

    fn accept_shadow_alpha(&self, intr: &SurfaceInteraction, ray: &Ray) -> bool {
        self.accept_mask(self.shadow_alpha_mask.as_ref(), intr, ray)
    }

    fn accept_mask(
        &self,
        source: Option<&AlphaMaskSource>,
        intr: &SurfaceInteraction,
        ray: &Ray,
    ) -> bool {
        let alpha = match source {
            Some(AlphaMaskSource::Texture(texture)) => {
                texture.evaluate(&TextureEvalContext::from(intr))
            }
            Some(AlphaMaskSource::Value(value)) => *value,
            None => return true,
        };

        if alpha <= 0.0 {
            return false;
        }
        if alpha >= 1.0 {
            return true;
        }
        hash_float_ray(ray) <= alpha
    }

    pub fn area(&self) -> Float {
        let shape = self.shape.as_ref();
        return shape.area();
    }

    pub fn pdf(&self, inter: &Interaction) -> Float {
        // Ignore any alpha textures used for trimming the shape when performing
        // this intersection. Hack for the "San Miguel" scene, where this is used
        // to make an invisible area light.
        let shape = self.shape.as_ref();
        return shape.pdf(inter);
    }

    pub fn pdf_from(&self, ctx: &ShapeSampleContext, wi: &Vector3f) -> Float {
        let shape = self.shape.as_ref();
        return shape.pdf_from(ctx, wi);
    }

    pub fn sample(&self, u: &Point2f) -> Option<(Interaction, Float)> {
        let shape = self.shape.as_ref();
        return shape.sample(u);
    }

    pub fn sample_from(
        &self,
        ctx: &ShapeSampleContext,
        u: &Point2f,
    ) -> Option<(Interaction, Float)> {
        let shape = self.shape.as_ref();
        return shape.sample_from(ctx, u);
    }

    pub fn solid_angle(&self, p: &Point3f, n_samples: i32) -> Float {
        let shape = self.shape.as_ref();
        return shape.solid_angle(p, n_samples);
    }
}

fn hash_float_ray(ray: &Ray) -> Float {
    let mut bytes = Vec::with_capacity(6 * std::mem::size_of::<Float>());
    for value in [ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z] {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    (murmur_hash_64a(&bytes, 0) as u32) as Float * (1.0 / 4_294_967_296.0)
}

fn murmur_hash_64a(key: &[u8], seed: u64) -> u64 {
    let m = 0xc6a4a7935bd1e995u64;
    let r = 47u32;
    let mut h = seed ^ (key.len() as u64).wrapping_mul(m);

    for block in key.chunks_exact(8) {
        let mut k = u64::from_ne_bytes(block.try_into().unwrap());
        k = k.wrapping_mul(m);
        k ^= k >> r;
        k = k.wrapping_mul(m);
        h ^= k;
        h = h.wrapping_mul(m);
    }

    let tail = key.chunks_exact(8).remainder();
    for (index, byte) in tail.iter().enumerate() {
        h ^= (*byte as u64) << (index * 8);
    }
    if !tail.is_empty() {
        h = h.wrapping_mul(m);
    }

    h ^= h >> r;
    h = h.wrapping_mul(m);
    h ^= h >> r;
    h
}
