use crate::util::base::*;

#[derive(Clone, Copy)]
pub struct BaseFilter {
    pub radius: Vector2f,
    pub inv_radius: Vector2f,
}

fn inverse_vector(radius: &Vector2f) -> Vector2f {
    Vector2f::new(1.0 / radius.x, 1.0 / radius.y)
}

impl BaseFilter {
    pub fn new(radius: &Vector2f) -> Self {
        BaseFilter {
            radius: *radius,
            inv_radius: inverse_vector(radius),
        }
    }
    pub fn get_radius(&self) -> Vector2f {
        self.radius
    }
    pub fn get_inv_radius(&self) -> Vector2f {
        self.inv_radius
    }
}
