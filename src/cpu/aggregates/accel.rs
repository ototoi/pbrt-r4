use crate::interaction::*;
use crate::util::base::*;
use crate::util::geometry::*;

use std::sync::Arc;

use super::bvh::accel::BVHAccel;
use super::exhaustive::ExhaustiveAccel;
use super::kdtree::KDTreeAccel;

/// Accel is an enum that holds different types of acceleration structures
#[derive(Clone)]
pub enum Accel {
    BVH(Arc<BVHAccel>),
    KdTree(Arc<KDTreeAccel>),
    Exhaustive(Arc<ExhaustiveAccel>),
}

impl Accel {
    /// Returns the world-space bounding box
    pub fn bounds(&self) -> Bounds3f {
        match self {
            Accel::BVH(a) => a.bounds(),
            Accel::KdTree(a) => a.bounds(),
            Accel::Exhaustive(a) => a.bounds(),
        }
    }

    /// Ray-acceleration structure intersection test
    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        match self {
            Accel::BVH(a) => a.intersect(r, t_max),
            Accel::KdTree(a) => a.intersect(r, t_max),
            Accel::Exhaustive(a) => a.intersect(r, t_max),
        }
    }

    /// Fast ray-acceleration structure intersection test
    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        match self {
            Accel::BVH(a) => a.intersect_p(r, t_max),
            Accel::KdTree(a) => a.intersect_p(r, t_max),
            Accel::Exhaustive(a) => a.intersect_p(r, t_max),
        }
    }
}
