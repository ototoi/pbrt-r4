#[derive(Clone, Debug, PartialEq)]
pub struct PerspectiveCamera {
    pub fov: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Camera {
    Perspective(PerspectiveCamera),
}
