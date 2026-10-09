use super::{Transform, Vertex};
use crate::util::error::PbrtError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SourceShape {
    #[default]
    TriangleMesh,
    PlyMesh,
    LoopSubdiv,
    Disk,
    Sphere,
    Cylinder,
    Cone,
    Paraboloid,
    HeightField,
    BilinearMesh,
    Hyperboloid,
    Nurbs,
}
impl SourceShape {
    pub fn from_name(name: &str) -> Result<Self, PbrtError> {
        match name {
            "trianglemesh" => Ok(Self::TriangleMesh),
            "plymesh" => Ok(Self::PlyMesh),
            "loopsubdiv" => Ok(Self::LoopSubdiv),
            "disk" => Ok(Self::Disk),
            "sphere" => Ok(Self::Sphere),
            "cylinder" => Ok(Self::Cylinder),
            "cone" => Ok(Self::Cone),
            "paraboloid" => Ok(Self::Paraboloid),
            "heightfield" => Ok(Self::HeightField),
            "bilinearmesh" => Ok(Self::BilinearMesh),
            "hyperboloid" => Ok(Self::Hyperboloid),
            "nurbs" => Ok(Self::Nurbs),
            _ => Err(PbrtError::error(&format!(
                "Unknown GPU source shape \"{name}\"."
            ))),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Geometry {
    pub source_shape: SourceShape,
    /// Range in `GeometryResources::vertices`.
    pub first_vertex: u32,
    pub vertex_count: u32,
    /// Range in `GeometryResources::indices`; each index addresses the full vertex array.
    pub first_index: u32,
    pub index_count: u32,
}

/// Packed vertex and index arrays with ranges describing each geometry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeometryResources {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub geometries: Vec<Geometry>,
}

pub fn triangle_area(positions: [[f32; 3]; 3]) -> f32 {
    let edge0 = sub3(positions[1], positions[0]);
    let edge1 = sub3(positions[2], positions[0]);
    let cross = [
        edge0[1] * edge1[2] - edge0[2] * edge1[1],
        edge0[2] * edge1[0] - edge0[0] * edge1[2],
        edge0[0] * edge1[1] - edge0[1] * edge1[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

pub fn triangle_geometric_normal(positions: [[f32; 3]; 3]) -> Result<[f32; 3], PbrtError> {
    let edge0 = sub3(positions[1], positions[0]);
    let edge1 = sub3(positions[2], positions[0]);
    let cross = [
        edge0[1] * edge1[2] - edge0[2] * edge1[1],
        edge0[2] * edge1[0] - edge0[0] * edge1[2],
        edge0[0] * edge1[1] - edge0[1] * edge1[0],
    ];
    let length = dot3(cross, cross).sqrt();
    if !length.is_finite() || length == 0.0 {
        return Err(PbrtError::error(
            "Area light triangle geometric normal is invalid.",
        ));
    }
    Ok(scale3(cross, 1.0 / length))
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn scale3(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

pub fn transform_point(matrix: &Transform, point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
        matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
        matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
    ]
}

pub fn transform_vector(matrix: &Transform, vector: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[4] * vector[0] + matrix[5] * vector[1] + matrix[6] * vector[2],
        matrix[8] * vector[0] + matrix[9] * vector[1] + matrix[10] * vector[2],
    ]
}
