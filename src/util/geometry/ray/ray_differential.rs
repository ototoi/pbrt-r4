use super::ray::Ray;
use crate::util::base::*;

#[derive(Default, Clone)]
pub struct RayDifferential {
    pub ray: Ray,
    pub has_differentials: bool,
    pub rx_origin: Point3f,
    pub ry_origin: Point3f,
    pub rx_direction: Vector3f,
    pub ry_direction: Vector3f,
}

impl RayDifferential {
    pub fn new(o: &Point3f, d: &Vector3f, t_max: Float, time: Float) -> Self {
        RayDifferential {
            ray: Ray::new(o, d, t_max, time),
            has_differentials: false,
            rx_origin: Point3f::default(),
            ry_origin: Point3f::default(),
            rx_direction: Vector3f::default(),
            ry_direction: Vector3f::default(),
        }
    }

    pub fn scale_differentials(&mut self, s: Float) {
        let rx_origin = self.ray.o + (self.rx_origin - self.ray.o) * s;
        let ry_origin = self.ray.o + (self.ry_origin - self.ray.o) * s;
        let rx_direction = self.ray.d + (self.rx_direction - self.ray.d) * s;
        let ry_direction = self.ray.d + (self.ry_direction - self.ray.d) * s;
        self.rx_origin = rx_origin;
        self.ry_origin = ry_origin;
        self.rx_direction = rx_direction;
        self.ry_direction = ry_direction;
    }
}

impl From<&Ray> for RayDifferential {
    fn from(ray: &Ray) -> Self {
        RayDifferential {
            ray: ray.clone(),
            has_differentials: false,
            rx_origin: Point3f::default(),
            ry_origin: Point3f::default(),
            rx_direction: Vector3f::default(),
            ry_direction: Vector3f::default(),
        }
    }
}

impl From<Ray> for RayDifferential {
    fn from(ray: Ray) -> Self {
        RayDifferential {
            ray,
            has_differentials: false,
            rx_origin: Point3f::default(),
            ry_origin: Point3f::default(),
            rx_direction: Vector3f::default(),
            ry_direction: Vector3f::default(),
        }
    }
}
