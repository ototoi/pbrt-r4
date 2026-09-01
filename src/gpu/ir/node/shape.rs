use super::types::Vec2f;
use super::types::Vec3f;

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleMeshShape {
    pub positions: Vec<Vec3f>,
    pub indices: Vec<u32>,
    pub normals: Option<Vec<Vec3f>>,
    pub tangents: Option<Vec<Vec3f>>,
    pub uvs: Option<Vec<Vec2f>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SphereShape {
    pub radius: f32,
    pub z_min: f32,
    pub z_max: f32,
    pub phi_max: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    TriangleMesh(Box<TriangleMeshShape>),
    Sphere(Box<SphereShape>),
}
