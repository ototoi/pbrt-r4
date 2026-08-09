use crate::media::*;
use crate::util::base::*;
use crate::util::geometry::*;

#[derive(Clone, Debug)]
pub struct BaseInteraction {
    pub p: Point3f,
    pub p_error: Vector3f,
    pub wo: Vector3f,
    pub n: Normal3f,
    pub uv: Point2f,
    pub time: Float,
    pub medium_interface: MediumInterface,
}

impl Default for BaseInteraction {
    fn default() -> Self {
        BaseInteraction {
            p: Point3f::zero(),
            p_error: Vector3f::zero(),
            wo: Vector3f::zero(),
            n: Normal3f::zero(),
            uv: Point2f::zero(),
            time: 0.0,
            medium_interface: MediumInterface::default(),
        }
    }
}

impl From<&Ray> for BaseInteraction {
    fn from(ray: &Ray) -> Self {
        BaseInteraction {
            p: ray.o,
            time: ray.time,
            p_error: Vector3f::zero(),
            n: Normal3f::zero(),
            wo: Vector3f::zero(),
            uv: Point2f::zero(),
            medium_interface: MediumInterface::from(&ray.medium),
        }
    }
}
