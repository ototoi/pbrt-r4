use super::Transform;

/// Shape-hit coverage, separate from the surface BSDF, as in pbrt-v4 primitives.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AlphaMask {
    #[default]
    Opaque,
    Constant(f32),
    /// Float root in `Scene.texture_library.roots`.
    Texture(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    /// Index into `Scene.geometry.geometries`.
    pub geometry: u32,
    pub transform: Transform,
    /// Index into `Scene.materials.roots`; `INVALID_INDEX` means the shape
    /// has no surface material (`Material ""`/`"interface"`).
    pub material_root: u32,
    pub alpha: AlphaMask,
    pub area_light: u32,
    pub reverse_orientation: bool,
    pub shape_transform_swaps_handedness: bool,
    /// Index into `Scene.media`; `INVALID_INDEX` means vacuum.
    pub inside_medium: u32,
    pub outside_medium: u32,
}
