use super::types::Vec2f;
use super::types::Vec3f;
use crate::paramdict::ParameterDictionary;
use crate::util::base::Float;
use crate::util::error::PbrtError;
use crate::util::mesh::TriQuadMesh;

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleMeshShape {
    pub positions: Vec<Vec3f>,
    pub indices: Vec<u32>,
    pub normals: Option<Vec<Vec3f>>,
    pub tangents: Option<Vec<Vec3f>>,
    pub uvs: Option<Vec<Vec2f>>,
}

#[derive(Clone)]
pub struct SphereShape {
    pub params: ParameterDictionary,
}

#[derive(Clone)]
pub enum Shape {
    TriangleMesh(Box<TriangleMeshShape>),
    Sphere(Box<SphereShape>),
}

pub fn triangle_mesh_from_params(
    shape_name: &str,
    params: &ParameterDictionary,
) -> Result<Option<TriangleMeshShape>, PbrtError> {
    if shape_name == "plymesh" {
        let filename = params.get_one_string("filename", "");
        let mut mesh = TriQuadMesh::read_ply(&filename)?;
        mesh.convert_to_only_triangles();
        return Ok(tri_quad_mesh_to_node_mesh(&mesh));
    }

    let points = params.get_points("P");
    if points.len() < 9 || points.len() % 3 != 0 {
        return Ok(None);
    }
    let positions = points
        .chunks_exact(3)
        .map(|p| Vec3f([p[0] as f32, p[1] as f32, p[2] as f32]))
        .collect::<Vec<_>>();
    let vertex_count = positions.len();
    let raw_indices = params.get_ints("indices");
    let indices = if raw_indices.is_empty() {
        if positions.len() == 3 {
            vec![0, 1, 2]
        } else {
            return Ok(None);
        }
    } else {
        if raw_indices.len() % 3 != 0 {
            return Ok(None);
        }
        let mut indices = Vec::with_capacity(raw_indices.len());
        for index in raw_indices {
            if index < 0 || index as usize >= positions.len() {
                return Ok(None);
            }
            indices.push(index as u32);
        }
        indices
    };

    Ok(Some(TriangleMeshShape {
        positions,
        indices,
        normals: node_vec3_attribute(params.get_points("N"), vertex_count),
        tangents: node_vec3_attribute(params.get_points("S"), vertex_count),
        uvs: node_vec2_attribute(params.get_points("uv"), vertex_count),
    }))
}

fn tri_quad_mesh_to_node_mesh(mesh: &TriQuadMesh) -> Option<TriangleMeshShape> {
    if mesh.p.is_empty() || mesh.tri_indices.is_empty() {
        return None;
    }
    let positions = mesh
        .p
        .iter()
        .map(|p| Vec3f([p.x as f32, p.y as f32, p.z as f32]))
        .collect::<Vec<_>>();
    let normals = if mesh.n.len() == positions.len() {
        Some(
            mesh.n
                .iter()
                .map(|n| Vec3f([n.x as f32, n.y as f32, n.z as f32]))
                .collect(),
        )
    } else {
        None
    };
    let uvs = if mesh.uv.len() == positions.len() {
        Some(
            mesh.uv
                .iter()
                .map(|uv| Vec2f([uv.x as f32, uv.y as f32]))
                .collect(),
        )
    } else {
        None
    };
    Some(TriangleMeshShape {
        positions,
        indices: mesh.tri_indices.clone(),
        normals,
        tangents: None,
        uvs,
    })
}

fn node_vec3_attribute(values: Vec<Float>, vertex_count: usize) -> Option<Vec<Vec3f>> {
    if values.is_empty() || values.len() != vertex_count * 3 {
        return None;
    }
    Some(
        values
            .chunks_exact(3)
            .map(|value| Vec3f([value[0] as f32, value[1] as f32, value[2] as f32]))
            .collect(),
    )
}

fn node_vec2_attribute(values: Vec<Float>, vertex_count: usize) -> Option<Vec<Vec2f>> {
    if values.is_empty() || values.len() != vertex_count * 2 {
        return None;
    }
    Some(
        values
            .chunks_exact(2)
            .map(|value| Vec2f([value[0] as f32, value[1] as f32]))
            .collect(),
    )
}
