use crate::base::light::Light;
use crate::base::material::*;
use crate::base::shape::Shape;
use crate::interaction::*;
use crate::media::*;
use crate::shapes::*;
use crate::util::geometry::*;
use crate::util::profile::*;

use std::sync::Arc;

#[derive(Clone)]
pub struct GeometricPrimitive {
    pub shape: Arc<Shape>,
    pub material: Option<Arc<Material>>,
    pub area_light: Option<Arc<Light>>,
    pub mi: MediumInterface,
}

impl GeometricPrimitive {
    pub fn new(
        shape: Arc<Shape>,
        material: &Option<Arc<Material>>,
        area_light: &Option<Arc<Light>>,
        mi: &MediumInterface,
    ) -> Self {
        GeometricPrimitive {
            shape,
            material: material.clone(),
            area_light: area_light.clone(),
            mi: mi.clone(),
        }
    }

    pub fn bounds(&self) -> Bounds3f {
        let s = self.shape.as_ref();
        s.world_bound()
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let _p = ProfilePhase::new(Prof::GeometricPrimitiveIntersect);

        let s = self.shape.as_ref();
        if let Some(mut si) = s.intersect(r, t_max) {
            si.intr.set_shape(&self.shape);
            si.intr.set_intersection_properties(
                &self.material,
                &self.area_light,
                Some(&self.mi),
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

    pub fn get_area_light(&self) -> Option<Arc<Light>> {
        self.area_light.clone()
    }

    pub fn get_material(&self) -> Option<Arc<Material>> {
        self.material.clone()
    }
}
