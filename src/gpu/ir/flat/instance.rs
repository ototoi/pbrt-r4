use super::Transform;

#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub geometry: u32,
    pub transform: Transform,
    pub material: u32,
}
