use super::camera::Camera;
use super::material::Material;
use super::shape::Shape;

use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub enum Component {
    Camera(Camera),
    Shape(Shape),
    Material(Arc<Material>),
}
