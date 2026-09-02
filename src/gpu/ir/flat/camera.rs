use super::Transform;

#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub camera_to_world: Transform,
    pub fov: f32,
    /// The pbrt screen window in the order [xmin, xmax, ymin, ymax].
    pub screen_window: [f32; 4],
}
