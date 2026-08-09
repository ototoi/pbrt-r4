use crate::interaction::*;
use crate::util::base::*;

/// ShapeIntersection holds information about a ray-shape intersection
///
/// This matches pbrt-v4's ShapeIntersection structure which combines
/// the surface interaction and the ray parameter t at the hit point.
#[derive(Clone, Debug)]
pub struct ShapeIntersection {
    /// Surface interaction at the intersection point
    pub intr: SurfaceInteraction,
    /// Ray parameter at the intersection point
    pub t_hit: Float,
}

impl ShapeIntersection {
    pub fn new(intr: SurfaceInteraction, t_hit: Float) -> Self {
        ShapeIntersection { intr, t_hit }
    }
}
