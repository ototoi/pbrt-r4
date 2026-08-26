// LightSampler interface, following pbrt-v4's Sample/PMF design.

use crate::base::light::Light;
use crate::base::light::{union_light_bounds, LightBounds};
use crate::cpu::integrators::IntegratorBase;
use crate::cpu::lightdistrib::lightdistrib::LightDistribution;
use crate::cpu::lightdistrib::power::compute_light_power_distribution;
use crate::cpu::lightdistrib::spatial::SpatialLightDistribution;
use crate::interaction::*;
use crate::util::base::*;
use crate::util::error::PbrtError;
use crate::util::geometry::{max_component, Bounds3f};
use crate::util::sampling::Distribution1D;
use crate::util::spectrum::SampledWavelengths;

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct SampledLight {
    pub light: Arc<Light>,
    pub p: Float,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LightSampleContext {
    pub p: Point3f,
    pub n: Normal3f,
    pub ns: Normal3f,
}

impl From<&Interaction> for LightSampleContext {
    fn from(value: &Interaction) -> Self {
        if let Some(si) = value.as_surface_interaction() {
            Self {
                p: si.p,
                n: si.n,
                ns: si.shading.n,
            }
        } else {
            Self {
                p: value.get_p(),
                n: Normal3f::zero(),
                ns: Normal3f::zero(),
            }
        }
    }
}

#[derive(Clone)]
pub struct UniformLightSampler {
    lights: Vec<Arc<Light>>,
}

impl UniformLightSampler {
    pub fn new(base: &IntegratorBase) -> Self {
        Self {
            lights: base.lights.clone(),
        }
    }

    pub fn sample(&self, u: Float) -> Option<SampledLight> {
        if self.lights.is_empty() {
            return None;
        }
        let light_index = usize::min(
            (u * self.lights.len() as Float) as usize,
            self.lights.len() - 1,
        );
        Some(SampledLight {
            light: self.lights[light_index].clone(),
            p: 1.0 / self.lights.len() as Float,
        })
    }

    pub fn pmf(&self, _light: &Arc<Light>) -> Float {
        if self.lights.is_empty() {
            0.0
        } else {
            1.0 / self.lights.len() as Float
        }
    }
}

#[derive(Clone)]
pub struct PowerLightSampler {
    lights: Vec<Arc<Light>>,
    light_to_index: HashMap<usize, usize>,
    distrib: Arc<Distribution1D>,
}

impl PowerLightSampler {
    pub fn new(base: &IntegratorBase) -> Self {
        let lights = base.lights.clone();
        let light_to_index = build_light_index_map(&lights);
        let distrib = if lights.is_empty() {
            Arc::new(Distribution1D::new(&[]))
        } else {
            compute_light_power_distribution(base)
        };
        Self {
            lights,
            light_to_index,
            distrib,
        }
    }

    pub fn sample(&self, u: Float) -> Option<SampledLight> {
        if self.lights.is_empty() {
            return None;
        }
        let (idx, p, _) = self.distrib.sample_discrete(u);
        if p <= 0.0 {
            return None;
        }
        Some(SampledLight {
            light: self.lights[idx].clone(),
            p,
        })
    }

    pub fn pmf(&self, light: &Arc<Light>) -> Float {
        if self.lights.is_empty() {
            return 0.0;
        }
        let key = light_ptr_key(light);
        if let Some(idx) = self.light_to_index.get(&key) {
            self.distrib.discrete_pdf(*idx)
        } else {
            0.0
        }
    }
}

#[derive(Clone)]
pub struct SpatialLightSampler {
    lights: Vec<Arc<Light>>,
    light_to_index: HashMap<usize, usize>,
    distrib: Arc<dyn LightDistribution>,
}

impl SpatialLightSampler {
    pub fn new(base: &IntegratorBase, max_voxels: usize) -> Self {
        let lights = base.lights.clone();
        let light_to_index = build_light_index_map(&lights);
        let distrib: Arc<dyn LightDistribution> =
            Arc::new(SpatialLightDistribution::new(base, max_voxels as u32));
        Self {
            lights,
            light_to_index,
            distrib,
        }
    }

    pub fn sample(&self, ctx: &LightSampleContext, u: Float) -> Option<SampledLight> {
        if self.lights.is_empty() {
            return None;
        }
        let d = self.distrib.lookup(&ctx.p);
        let (idx, p, _) = d.sample_discrete(u);
        if p <= 0.0 {
            return None;
        }
        Some(SampledLight {
            light: self.lights[idx].clone(),
            p,
        })
    }

    pub fn pmf(&self, ctx: &LightSampleContext, light: &Arc<Light>) -> Float {
        if self.lights.is_empty() {
            return 0.0;
        }
        let key = light_ptr_key(light);
        if let Some(idx) = self.light_to_index.get(&key) {
            self.distrib.lookup(&ctx.p).discrete_pdf(*idx)
        } else {
            0.0
        }
    }
}

/// pbrt-v4 verbatim `BVHLightSampler` (`lightsamplers.h:259-358`).
///
/// Sampling proceeds in two stages:
///   * with probability `pInfinite = n_infinite / (n_infinite + has_bounded)`,
///     pick uniformly among the infinite (env / portal) lights;
///   * otherwise traverse a Light-BVH built over the bounded lights and
///     at every interior node pick a child weighted by
///     `LightBounds::Importance(p, n)`.
///
/// The BVH is built once at construction time via a 12-bucket SAH
/// (matching v4 `BVHLightSampler::buildBVH`). `CompactLightBounds`
/// (v4's octahedral-encoded quantisation) is intentionally not ported;
/// r4 stores the plain `LightBounds` on each node so the importance
/// calculation can use full-precision data with no extra decoding step.
#[derive(Clone, Debug)]
enum LightBVHNode {
    Leaf {
        light_index: usize,
        bounds: LightBounds,
    },
    Interior {
        /// Index of the right child in the `nodes` vector. The left
        /// child is always at `parent_index + 1` (DFS order).
        right_child: usize,
        bounds: LightBounds,
    },
}

impl LightBVHNode {
    fn bounds(&self) -> &LightBounds {
        match self {
            LightBVHNode::Leaf { bounds, .. } => bounds,
            LightBVHNode::Interior { bounds, .. } => bounds,
        }
    }
}

#[derive(Clone)]
pub struct BVHLightSampler {
    lights: Vec<Arc<Light>>,
    light_to_index: HashMap<usize, usize>,
    infinite_lights: Vec<Arc<Light>>,
    /// DFS-ordered BVH nodes. Empty when there are no bounded lights.
    nodes: Vec<LightBVHNode>,
    /// `lightToBitTrail` in v4: for every bounded light, encodes the
    /// path from the root to its leaf as a sequence of left(0)/right(1)
    /// child choices.
    light_to_bit_trail: HashMap<usize, u32>,
}

impl BVHLightSampler {
    pub fn new(base: &IntegratorBase, _max_voxels: usize) -> Self {
        Self::from_lights(base.lights.clone())
    }

    /// Construct the sampler from the light list accepted by the v4
    /// `BVHLightSampler` constructor.
    pub fn from_lights(lights: Vec<Arc<Light>>) -> Self {
        let light_to_index = build_light_index_map(&lights);
        let mut infinite_lights: Vec<Arc<Light>> = Vec::new();
        let mut bvh_lights: Vec<(usize, LightBounds)> = Vec::new();
        for (i, light) in lights.iter().enumerate() {
            match light.bounds() {
                None => infinite_lights.push(light.clone()),
                Some(lb) if lb.phi > 0.0 => bvh_lights.push((i, lb)),
                _ => {}
            }
        }
        let mut sampler = Self {
            lights,
            light_to_index,
            infinite_lights,
            nodes: Vec::new(),
            light_to_bit_trail: HashMap::new(),
        };
        if !bvh_lights.is_empty() {
            let end = bvh_lights.len();
            sampler.build_bvh(&mut bvh_lights, 0, end, 0, 0);
        }
        sampler
    }

    fn p_infinite(&self) -> Float {
        let n_inf = self.infinite_lights.len();
        let bounded_bucket = if self.nodes.is_empty() { 0 } else { 1 };
        if n_inf + bounded_bucket == 0 {
            0.0
        } else {
            (n_inf as Float) / ((n_inf + bounded_bucket) as Float)
        }
    }

    /// pbrt-v4 `BVHLightSampler::buildBVH` (lightsamplers.cpp:135).
    /// Recursive 12-bucket SAH split on the LightBounds centroids;
    /// returns `(node_index, combined_LightBounds)` for the subtree.
    fn build_bvh(
        &mut self,
        bvh_lights: &mut Vec<(usize, LightBounds)>,
        start: usize,
        end: usize,
        bit_trail: u32,
        depth: u32,
    ) -> (usize, LightBounds) {
        debug_assert!(start < end);
        if end - start == 1 {
            let node_index = self.nodes.len();
            let (light_index, ref bounds) = bvh_lights[start];
            self.nodes.push(LightBVHNode::Leaf {
                light_index,
                bounds: bounds.clone(),
            });
            self.light_to_bit_trail
                .insert(light_index_key(&self.lights[light_index]), bit_trail);
            return (node_index, bounds.clone());
        }

        // Compute centroid bounds for split-axis selection.
        let mut centroid_bounds = Bounds3f::default();
        let mut combined_bounds = Bounds3f::default();
        for (_, lb) in bvh_lights[start..end].iter() {
            centroid_bounds = centroid_bounds.union_p(&lb.centroid());
            combined_bounds = combined_bounds.union(&lb.bounds);
        }

        const N_BUCKETS: usize = 12;
        let mut min_cost = Float::INFINITY;
        let mut min_bucket: i32 = -1;
        let mut min_dim: i32 = -1;
        for dim in 0..3 {
            let lo = centroid_bounds.min[dim];
            let hi = centroid_bounds.max[dim];
            if hi == lo {
                continue;
            }
            let mut bucket_bounds: [Option<LightBounds>; N_BUCKETS] = Default::default();
            for (_, lb) in bvh_lights[start..end].iter() {
                let pc = lb.centroid()[dim];
                let mut b = ((pc - lo) / (hi - lo) * N_BUCKETS as Float) as i32;
                if b >= N_BUCKETS as i32 {
                    b = N_BUCKETS as i32 - 1;
                }
                let bi = b.max(0) as usize;
                bucket_bounds[bi] = Some(match &bucket_bounds[bi] {
                    Some(existing) => union_light_bounds(existing, lb),
                    None => lb.clone(),
                });
            }

            for i in 1..(N_BUCKETS - 1) {
                let mut b0: Option<LightBounds> = None;
                let mut b1: Option<LightBounds> = None;
                for j in 0..=i {
                    if let Some(bb) = bucket_bounds[j].as_ref() {
                        b0 = Some(match b0 {
                            Some(existing) => union_light_bounds(&existing, bb),
                            None => bb.clone(),
                        });
                    }
                }
                for j in (i + 1)..N_BUCKETS {
                    if let Some(bb) = bucket_bounds[j].as_ref() {
                        b1 = Some(match b1 {
                            Some(existing) => union_light_bounds(&existing, bb),
                            None => bb.clone(),
                        });
                    }
                }
                let cost = evaluate_cost(b0.as_ref(), &combined_bounds, dim)
                    + evaluate_cost(b1.as_ref(), &combined_bounds, dim);
                if cost > 0.0 && cost < min_cost {
                    min_cost = cost;
                    min_bucket = i as i32;
                    min_dim = dim as i32;
                }
            }
        }

        let mid = if min_dim < 0 {
            (start + end) / 2
        } else {
            let dim = min_dim as usize;
            let lo = centroid_bounds.min[dim];
            let hi = centroid_bounds.max[dim];
            let mut left = start;
            let mut right = end;
            while left < right {
                let pc = bvh_lights[left].1.centroid()[dim];
                let mut b = ((pc - lo) / (hi - lo) * N_BUCKETS as Float) as i32;
                if b >= N_BUCKETS as i32 {
                    b = N_BUCKETS as i32 - 1;
                }
                if b <= min_bucket {
                    left += 1;
                } else {
                    right -= 1;
                    bvh_lights.swap(left, right);
                }
            }
            let mut split = left;
            if split == start || split == end {
                split = (start + end) / 2;
            }
            split
        };

        let node_index = self.nodes.len();
        // Reserve interior slot; children DFS-append after it.
        self.nodes.push(LightBVHNode::Leaf {
            light_index: 0,
            bounds: LightBounds {
                bounds: Bounds3f::default(),
                w: Vector3f::new(0.0, 0.0, 1.0),
                phi: 0.0,
                cos_theta_o: 1.0,
                cos_theta_e: 1.0,
                two_sided: false,
            },
        });
        debug_assert!(depth < 64);
        let (left_idx, b0) = self.build_bvh(bvh_lights, start, mid, bit_trail, depth + 1);
        debug_assert_eq!(node_index + 1, left_idx);
        let (right_idx, b1) =
            self.build_bvh(bvh_lights, mid, end, bit_trail | (1u32 << depth), depth + 1);
        let combined = union_light_bounds(&b0, &b1);
        self.nodes[node_index] = LightBVHNode::Interior {
            right_child: right_idx,
            bounds: combined.clone(),
        };
        (node_index, combined)
    }

    pub fn sample(&self, ctx: &LightSampleContext, u: Float) -> Option<SampledLight> {
        if self.lights.is_empty() {
            return None;
        }
        let p_inf = self.p_infinite();
        if u < p_inf {
            let n = self.infinite_lights.len();
            if n == 0 {
                return None;
            }
            let u_inf = u / p_inf;
            let i = usize::min((u_inf * n as Float) as usize, n - 1);
            let pmf = p_inf / n as Float;
            return Some(SampledLight {
                light: self.infinite_lights[i].clone(),
                p: pmf,
            });
        }
        if self.nodes.is_empty() {
            return None;
        }
        let mut u = ((u - p_inf) / (1.0 - p_inf)).min(1.0 - 1e-6);
        let mut node_index = 0usize;
        let mut pmf = 1.0 - p_inf;
        let p = ctx.p;
        let n = ctx.ns;
        loop {
            match &self.nodes[node_index] {
                LightBVHNode::Leaf { light_index, .. } => {
                    return Some(SampledLight {
                        light: self.lights[*light_index].clone(),
                        p: pmf,
                    });
                }
                LightBVHNode::Interior { right_child, .. } => {
                    let left_idx = node_index + 1;
                    let right_idx = *right_child;
                    let ci0 = self.nodes[left_idx].bounds().importance(p, n);
                    let ci1 = self.nodes[right_idx].bounds().importance(p, n);
                    if ci0 == 0.0 && ci1 == 0.0 {
                        return None;
                    }
                    let total = ci0 + ci1;
                    // Sample-discrete on a two-element distribution.
                    let p0 = ci0 / total;
                    if u < p0 {
                        u = (u / p0).min(1.0 - 1e-6);
                        pmf *= p0;
                        node_index = left_idx;
                    } else {
                        u = ((u - p0) / (1.0 - p0)).min(1.0 - 1e-6);
                        pmf *= 1.0 - p0;
                        node_index = right_idx;
                    }
                }
            }
        }
    }

    pub fn pmf(&self, ctx: &LightSampleContext, light: &Arc<Light>) -> Float {
        if self.lights.is_empty() {
            return 0.0;
        }
        let key = light_ptr_key(light);
        let Some(&idx) = self.light_to_index.get(&key) else {
            return 0.0;
        };
        let p_inf = self.p_infinite();
        if self.lights[idx].bounds().is_none() {
            if self.infinite_lights.is_empty() {
                return 0.0;
            }
            return p_inf / self.infinite_lights.len() as Float;
        }
        if self.nodes.is_empty() {
            return 0.0;
        }
        let bit_trail = match self.light_to_bit_trail.get(&key) {
            Some(&t) => t,
            None => return 0.0,
        };
        let mut pmf = 1.0 - p_inf;
        let mut node_index = 0usize;
        let mut trail = bit_trail;
        let p = ctx.p;
        let n = ctx.ns;
        loop {
            match &self.nodes[node_index] {
                LightBVHNode::Leaf { .. } => return pmf,
                LightBVHNode::Interior { right_child, .. } => {
                    let left_idx = node_index + 1;
                    let right_idx = *right_child;
                    let ci0 = self.nodes[left_idx].bounds().importance(p, n);
                    let ci1 = self.nodes[right_idx].bounds().importance(p, n);
                    let total = ci0 + ci1;
                    if total == 0.0 {
                        return 0.0;
                    }
                    if trail & 1 == 0 {
                        pmf *= ci0 / total;
                        node_index = left_idx;
                    } else {
                        pmf *= ci1 / total;
                        node_index = right_idx;
                    }
                    trail >>= 1;
                }
            }
        }
    }
}

fn light_index_key(light: &Arc<Light>) -> usize {
    light_ptr_key(light)
}

fn evaluate_cost(b: Option<&LightBounds>, parent: &Bounds3f, dim: usize) -> Float {
    let Some(lb) = b else { return 0.0 };
    // Evaluate direction bounds measure for LightBounds
    let theta_o = lb.cos_theta_o.acos();
    let theta_w = (theta_o + lb.cos_theta_e.acos()).min(std::f32::consts::PI as Float);
    let sin_theta_o = Float::sqrt(Float::max(0.0, 1.0 - lb.cos_theta_o * lb.cos_theta_o));
    let m_omega = 2.0 * std::f32::consts::PI as Float * (1.0 - lb.cos_theta_o)
        + std::f32::consts::PI as Float / 2.0
            * (2.0 * theta_w * sin_theta_o
                - (theta_o - 2.0 * theta_w).cos()
                - 2.0 * theta_o * sin_theta_o
                + lb.cos_theta_o);

    let diagonal = parent.diagonal();
    let kr = max_component(&diagonal) / diagonal[dim];
    lb.phi * m_omega * kr * lb.bounds.surface_area()
}

#[derive(Clone)]
pub struct ExhaustiveLightSampler {
    lights: Vec<Arc<Light>>,
    light_to_index: HashMap<usize, usize>,
    infinite_indices: Vec<usize>,
    bounded_indices: Vec<usize>,
}

impl ExhaustiveLightSampler {
    pub fn new(base: &IntegratorBase) -> Self {
        let lights = base.lights.clone();
        let light_to_index = build_light_index_map(&lights);
        let mut infinite_indices = Vec::new();
        let mut bounded_indices = Vec::new();
        for (i, light) in lights.iter().enumerate() {
            if light.is_infinite() {
                infinite_indices.push(i);
            } else {
                bounded_indices.push(i);
            }
        }
        Self {
            lights,
            light_to_index,
            infinite_indices,
            bounded_indices,
        }
    }

    fn p_infinite(&self) -> Float {
        if self.lights.is_empty() {
            return 0.0;
        }
        let n_infinite = self.infinite_indices.len();
        let bounded_bucket = if self.bounded_indices.is_empty() {
            0
        } else {
            1
        };
        (n_infinite as Float) / ((n_infinite + bounded_bucket) as Float)
    }

    fn bounded_importance(&self, light_index: usize, ctx: &LightSampleContext) -> Float {
        let light = &self.lights[light_index];
        let u = Point2f::new(0.5, 0.5);
        // Materialize a default lambda for the importance probe; the
        // result is averaged before being compared so wavelength choice
        // only matters when the light has strong spectral variation.
        let lambda = SampledWavelengths::sample_visible(0.5);
        if let Some(s) = light.sample_li(ctx, u, &lambda, false) {
            if s.pdf > 0.0 && !s.l.is_black() {
                return Float::max(0.0, s.l.average() / s.pdf);
            }
        }
        Float::max(0.0, light.phi(&lambda).average())
    }

    pub fn sample(&self, ctx: &LightSampleContext, u: Float) -> Option<SampledLight> {
        if self.lights.is_empty() {
            return None;
        }
        let p_infinite = self.p_infinite();
        if u < p_infinite {
            if self.infinite_indices.is_empty() {
                return None;
            }
            let u_inf = u / p_infinite;
            let i = usize::min(
                (u_inf * self.infinite_indices.len() as Float) as usize,
                self.infinite_indices.len() - 1,
            );
            let light_index = self.infinite_indices[i];
            return Some(SampledLight {
                light: self.lights[light_index].clone(),
                p: p_infinite / self.infinite_indices.len() as Float,
            });
        }

        if self.bounded_indices.is_empty() {
            return None;
        }
        let u_bounded = Float::min((u - p_infinite) / (1.0 - p_infinite), 1.0 - 1e-6);

        let mut weights = Vec::with_capacity(self.bounded_indices.len());
        let mut sum = 0.0;
        for &idx in self.bounded_indices.iter() {
            let w = self.bounded_importance(idx, ctx);
            weights.push(w);
            sum += w;
        }

        if sum <= 0.0 {
            let i = usize::min(
                (u_bounded * self.bounded_indices.len() as Float) as usize,
                self.bounded_indices.len() - 1,
            );
            let light_index = self.bounded_indices[i];
            let p = (1.0 - p_infinite) / self.bounded_indices.len() as Float;
            return Some(SampledLight {
                light: self.lights[light_index].clone(),
                p,
            });
        }

        let target = u_bounded * sum;
        let mut accum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            accum += w;
            if target <= accum {
                let light_index = self.bounded_indices[i];
                let p = (1.0 - p_infinite) * w / sum;
                return Some(SampledLight {
                    light: self.lights[light_index].clone(),
                    p,
                });
            }
        }

        let light_index = *self.bounded_indices.last().unwrap();
        let w = *weights.last().unwrap();
        Some(SampledLight {
            light: self.lights[light_index].clone(),
            p: (1.0 - p_infinite) * w / sum,
        })
    }

    pub fn pmf(&self, ctx: &LightSampleContext, light: &Arc<Light>) -> Float {
        if self.lights.is_empty() {
            return 0.0;
        }
        let key = light_ptr_key(light);
        let Some(&idx) = self.light_to_index.get(&key) else {
            return 0.0;
        };
        let p_infinite = self.p_infinite();

        if self.lights[idx].is_infinite() {
            if self.infinite_indices.is_empty() {
                return 0.0;
            }
            return p_infinite / self.infinite_indices.len() as Float;
        }

        if self.bounded_indices.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut w_light = 0.0;
        for &j in self.bounded_indices.iter() {
            let w = self.bounded_importance(j, ctx);
            if j == idx {
                w_light = w;
            }
            sum += w;
        }
        if sum <= 0.0 {
            return (1.0 - p_infinite) / self.bounded_indices.len() as Float;
        }
        (1.0 - p_infinite) * w_light / sum
    }
}

#[derive(Clone)]
pub enum LightSampler {
    Uniform(UniformLightSampler),
    Power(PowerLightSampler),
    Exhaustive(ExhaustiveLightSampler),
    BVH(BVHLightSampler),
    Spatial(SpatialLightSampler),
}

impl LightSampler {
    pub fn create(name: &str, base: &IntegratorBase) -> Result<Self, PbrtError> {
        let sampler = match name {
            "uniform" => Self::Uniform(UniformLightSampler::new(base)),
            "power" => Self::Power(PowerLightSampler::new(base)),
            "exhaustive" => Self::Exhaustive(ExhaustiveLightSampler::new(base)),
            "bvh" => Self::BVH(BVHLightSampler::new(base, 64)),
            "spatial" => Self::Spatial(SpatialLightSampler::new(base, 64)),
            _ => {
                return Err(PbrtError::error(&format!(
                    "Light sampler strategy \"{}\" unknown.",
                    name
                )));
            }
        };
        Ok(sampler)
    }

    pub fn sample(&self, ctx: &LightSampleContext, u: Float) -> Option<SampledLight> {
        match self {
            Self::Uniform(s) => s.sample(u),
            Self::Power(s) => s.sample(u),
            Self::Exhaustive(s) => s.sample(ctx, u),
            Self::BVH(s) => s.sample(ctx, u),
            Self::Spatial(s) => s.sample(ctx, u),
        }
    }

    pub fn pmf(&self, ctx: &LightSampleContext, light: &Arc<Light>) -> Float {
        match self {
            Self::Uniform(s) => s.pmf(light),
            Self::Power(s) => s.pmf(light),
            Self::Exhaustive(s) => s.pmf(ctx, light),
            Self::BVH(s) => s.pmf(ctx, light),
            Self::Spatial(s) => s.pmf(ctx, light),
        }
    }
}

fn light_ptr_key(light: &Arc<Light>) -> usize {
    Arc::as_ptr(light) as *const () as usize
}

fn build_light_index_map(lights: &[Arc<Light>]) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    for (i, light) in lights.iter().enumerate() {
        map.insert(light_ptr_key(light), i);
    }
    map
}
