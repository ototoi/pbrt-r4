use super::Transform;

#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub geometry: u32,
    pub transform: Transform,
    pub material: u32,
    pub first_area_light: u32,
    pub reverse_orientation: bool,
}
