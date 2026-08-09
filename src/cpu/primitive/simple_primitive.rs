use crate::base::material::*;
use crate::base::shape::Shape;
use crate::interaction::*;
use crate::shapes::*;
use crate::util::geometry::*;
use crate::util::profile::*;

use std::sync::Arc;

/// SimplePrimitive represents a shape with a material but no area light or medium interface
#[derive(Clone)]
pub struct SimplePrimitive {
    pub shape: Arc<Shape>,
    pub material: Arc<Material>,
}

impl SimplePrimitive {
    pub fn new(shape: Arc<Shape>, material: Arc<Material>) -> Self {
        SimplePrimitive { shape, material }
    }

    pub fn bounds(&self) -> Bounds3f {
        self.shape.as_ref().world_bound()
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let _p = ProfilePhase::new(Prof::GeometricPrimitiveIntersect);

        let s = self.shape.as_ref();
        if let Some(mut si) = s.intersect(r, t_max) {
            si.intr.set_shape(&self.shape);
            si.intr.set_intersection_properties(
                &Some(self.material.clone()),
                &None,
                None,
                &r.medium,
            );
            return Some(si);
        }
        None
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        let _p = ProfilePhase::new(Prof::GeometricPrimitiveIntersectP);

        let s = self.shape.as_ref();
        s.intersect_p(r, t_max)
    }
}
