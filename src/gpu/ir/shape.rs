use super::types::Vec2;
use super::types::Vec3;

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleMeshShape {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub normals: Option<Vec<Vec3>>,
    pub tangents: Option<Vec<Vec3>>,
    pub uvs: Option<Vec<Vec2>>,
    pub face_indices: Option<Vec<u32>>,
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
    TriangleMesh(TriangleMeshShape),
    Sphere(SphereShape),
}
