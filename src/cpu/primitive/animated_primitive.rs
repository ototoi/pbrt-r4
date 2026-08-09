use crate::interaction::*;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::transform::*;

use std::sync::Arc;

use super::Primitive;

/// AnimatedPrimitive wraps a primitive with an animated transformation
#[derive(Clone)]
pub struct AnimatedPrimitive {
    primitive: Arc<Primitive>,
    render_from_primitive: Box<AnimatedTransform>,
}

impl AnimatedPrimitive {
    pub fn new(primitive: &Arc<Primitive>, render_from_primitive: &AnimatedTransform) -> Self {
        AnimatedPrimitive {
            primitive: primitive.clone(),
            render_from_primitive: Box::new(render_from_primitive.clone()),
        }
    }

    pub fn bounds(&self) -> Bounds3f {
        let b = self.primitive.bounds();
        self.render_from_primitive.motion_bounds(&b)
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let m = self.render_from_primitive.interpolate(r.time);
        let (ray, _, _) = m.inverse().transform_ray(r);

        if let Some(mut si) = self.primitive.intersect(&ray, t_max) {
            si.intr = m.transform_surface_interaction(&si.intr);
            return Some(si);
        }
        None
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        let m = self.render_from_primitive.interpolate(r.time);
        let (ray, _, _) = m.inverse().transform_ray(r);
        self.primitive.intersect_p(&ray, t_max)
    }
}
