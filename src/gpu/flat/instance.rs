use super::Transform;

#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub geometry: u32,
    pub transform: Transform,
    /// Index into `Scene.material_roots`; `INVALID_INDEX` means the shape
    /// has no surface material (`Material ""`/`"interface"`).
    pub material_root: u32,
    pub area_light: u32,
    pub reverse_orientation: bool,
    /// Index into `Scene.media`; `INVALID_INDEX` means vacuum.
    pub inside_medium: u32,
    pub outside_medium: u32,
}
