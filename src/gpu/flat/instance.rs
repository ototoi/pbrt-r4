use super::Transform;

#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub geometry: u32,
    pub transform: Transform,
    pub material_root: u32,
    pub area_light: u32,
    pub reverse_orientation: bool,
    /// Index into `Scene.media`; `INVALID_INDEX` means vacuum.
    pub inside_medium: u32,
    pub outside_medium: u32,
}
