use crate::interaction::*;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::transform::*;

use std::sync::Arc;

use super::Primitive;

#[derive(Clone)]
pub struct TransformedPrimitive {
    primitive: Arc<Primitive>,
    primitive_to_world: Box<AnimatedTransform>,
}

impl TransformedPrimitive {
    pub fn new(primitive: &Arc<Primitive>, primitive_to_world: &AnimatedTransform) -> Self {
        TransformedPrimitive {
            primitive: primitive.clone(),
            primitive_to_world: Box::new(primitive_to_world.clone()),
        }
    }

    pub fn bounds(&self) -> Bounds3f {
        let b = self.primitive.bounds();
        self.primitive_to_world.motion_bounds(&b)
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let m = self.primitive_to_world.interpolate(r.time);
        let (ray, _, _) = m.inverse().transform_ray(r);

        if let Some(mut si) = self.primitive.intersect(&ray, t_max) {
            si.intr = m.transform_surface_interaction(&si.intr);
            return Some(si);
        }
        None
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        let m = self.primitive_to_world.interpolate(r.time);
        let (ray, _, _) = m.inverse().transform_ray(r);
        self.primitive.intersect_p(&ray, t_max)
    }
}
