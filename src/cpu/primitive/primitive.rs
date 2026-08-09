use crate::base::light::Light;
use crate::base::material::*;
use crate::base::shape::Shape;
use crate::interaction::*;
use crate::media::medium_interface::MediumInterface;
use crate::util::base::*;
use crate::util::geometry::*;

use std::sync::Arc;

use super::animated_primitive::AnimatedPrimitive;
use super::geometric_primitive::GeometricPrimitive;
use super::simple_primitive::SimplePrimitive;
use super::transformed_primitive::TransformedPrimitive;

use crate::cpu::aggregates::Accel;

/// Primitive is an enum that can hold different types of primitives
/// This follows pbrt-v4's TaggedPointer approach
#[derive(Clone)]
pub enum Primitive {
    Simple(SimplePrimitive),
    Geometric(Box<GeometricPrimitive>),
    Transformed(TransformedPrimitive),
    Animated(AnimatedPrimitive),
    Accel(Accel),
}

impl Primitive {
    pub fn new_geometric(
        shape: Arc<Shape>,
        material: &Option<Arc<Material>>,
        area_light: &Option<Arc<Light>>,
        mi: &MediumInterface,
    ) -> Self {
        // Triangle-heavy scenes should not pay for area light / medium
        // fields on every primitive when neither is active.
        if area_light.is_none() && !mi.is_medium_transition() {
            if let Some(material) = material {
                return Primitive::Simple(SimplePrimitive::new(shape, Arc::clone(material)));
            }
        }
        Primitive::Geometric(Box::new(GeometricPrimitive::new(
            shape, material, area_light, mi,
        )))
    }

    /// Returns the world-space bounding box
    pub fn bounds(&self) -> Bounds3f {
        match self {
            Primitive::Simple(p) => p.bounds(),
            Primitive::Geometric(p) => p.bounds(),
            Primitive::Transformed(p) => p.bounds(),
            Primitive::Animated(p) => p.bounds(),
            Primitive::Accel(a) => a.bounds(),
        }
    }

    /// Ray-primitive intersection test
    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        match self {
            Primitive::Simple(p) => p.intersect(r, t_max),
            Primitive::Geometric(p) => p.intersect(r, t_max),
            Primitive::Transformed(p) => p.intersect(r, t_max),
            Primitive::Animated(p) => p.intersect(r, t_max),
            Primitive::Accel(a) => a.intersect(r, t_max),
        }
    }

    /// Fast ray-primitive intersection test (no surface interaction computed)
    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        match self {
            Primitive::Simple(p) => p.intersect_p(r, t_max),
            Primitive::Geometric(p) => p.intersect_p(r, t_max),
            Primitive::Transformed(p) => p.intersect_p(r, t_max),
            Primitive::Animated(p) => p.intersect_p(r, t_max),
            Primitive::Accel(a) => a.intersect_p(r, t_max),
        }
    }

    /// Returns the area light if this primitive is emissive
    pub fn get_area_light(&self) -> Option<Arc<Light>> {
        match self {
            Primitive::Geometric(p) => p.get_area_light(),
            _ => None,
        }
    }

    /// Returns the material
    pub fn get_material(&self) -> Option<Arc<Material>> {
        match self {
            Primitive::Simple(p) => Some(p.material.clone()),
            Primitive::Geometric(p) => p.get_material(),
            _ => None,
        }
    }

    /// Returns true if this is a geometric primitive
    pub fn is_geometric(&self) -> bool {
        matches!(self, Primitive::Simple(_) | Primitive::Geometric(_))
    }
}
