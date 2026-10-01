#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    /// Object-space normal with the shape's ReverseOrientation applied.
    /// The instance normal transform is applied by the shader; zero means absent.
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub uv: [f32; 2],
}
