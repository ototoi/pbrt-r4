pub use crate::util::base::*;
pub use crate::util::transform::*;

#[derive(Debug, PartialEq, Clone)]
pub struct BaseShape {
    pub object_to_world: Transform,
    pub world_to_object: Transform,
    pub reverse_orientation: bool,
    pub swaps_handedness: bool,
}

impl BaseShape {
    pub fn new(
        object_to_world: &Transform,
        world_to_object: &Transform,
        reverse_orientation: bool,
    ) -> Self {
        let swaps_handedness = object_to_world.swaps_handedness();
        BaseShape {
            object_to_world: *object_to_world,
            world_to_object: *world_to_object,
            reverse_orientation,
            swaps_handedness,
        }
    }

    pub fn calc_normal(&self, dpdu: &Vector3f, dpdv: &Vector3f) -> Normal3f {
        let mut n = Vector3f::cross(dpdu, dpdv).normalize();
        if self.reverse_orientation ^ self.swaps_handedness {
            n *= -1.0;
        }
        return n;
    }
}
