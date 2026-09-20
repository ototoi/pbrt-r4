use super::Transform;

#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub geometry: u32,
    pub transform: Transform,
    pub material_tree_layout: u32,
    pub area_light: u32,
    pub reverse_orientation: bool,
}
