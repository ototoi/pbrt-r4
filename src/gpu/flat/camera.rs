use super::Transform;

#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub camera_to_world: Transform,
    pub kind: String,
    pub fov: f32,
    pub disable_texture_filtering: bool,
    pub disable_pixel_jitter: bool,
    /// The pbrt screen window in the order [xmin, xmax, ymin, ymax].
    pub screen_window: [f32; 4],
    /// Index into `Scene.media`; `INVALID_INDEX` means vacuum.
    pub medium: u32,
}
