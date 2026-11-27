//! CPU acceleration for a group of curves sharing primitive-level properties.
use crate::base::material::Material;
use crate::cpu::primitive::Primitive;
use crate::interaction::ShapeIntersection;
use crate::media::medium_interface::MediumInterface;
use crate::shapes::curves::CurvesShape;
use crate::util::base::Float;
use crate::util::geometry::{Bounds3f, Ray};
use std::sync::Arc;

use super::Accel;

pub struct CurvesAccel {
    shape: CurvesShape,
    material: Option<Arc<Material>>,
    medium: MediumInterface,
}

impl CurvesAccel {
    pub fn new(
        shape: CurvesShape,
        material: Option<Arc<Material>>,
        medium: MediumInterface,
    ) -> Self {
        Self {
            shape,
            material,
            medium,
        }
    }

    pub fn bounds(&self) -> Bounds3f {
        self.shape.world_bound()
    }

    pub fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        self.shape.intersect(ray, t_max).map(|mut hit| {
            hit.intr.set_intersection_properties(
                &self.material,
                &None,
                Some(&self.medium),
                &ray.medium,
            );
            hit
        })
    }

    pub fn intersect_p(&self, ray: &Ray, t_max: Float) -> bool {
        self.shape.intersect_p(ray, t_max)
    }
}

impl From<CurvesAccel> for Primitive {
    fn from(accel: CurvesAccel) -> Self {
        Self::Accel(Accel::Curves(Arc::new(accel)))
    }
}
