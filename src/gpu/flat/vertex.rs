#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    /// Object-space normal with ReverseOrientation applied; zero means absent.
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub uv: [f32; 2],
}
