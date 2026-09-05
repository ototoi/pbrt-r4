use super::types::Vec2f;
use super::types::Vec3f;
use crate::base::shape::Shape as CpuShape;
use crate::paramdict::ParameterDictionary;
use crate::shapes::LoopSubdiv;
use crate::util::error::PbrtError;
use crate::util::mesh::TriQuadMesh;
use crate::util::transform::Transform as CpuTransform;

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
        normals: node_vec3_attribute(params, "N", vertex_count)?,
        tangents: node_vec3_attribute(params, "S", vertex_count)?,
        uvs: node_vec2_attribute(params, "uv", vertex_count)?,
    }))
}

/// Converts the CPU Loop Subdivision result into the Node IR mesh form.
///
/// The CPU implementation creates one `Triangle` value per output face, but
/// all of those values share a single `TriangleMesh`. The GPU IR stores that
/// shared mesh once instead of creating one Node IR shape per face.
pub fn loop_subdiv_mesh_from_params(
    params: &ParameterDictionary,
    reverse_orientation: bool,
) -> Result<Option<TriangleMeshShape>, PbrtError> {
    let identity = CpuTransform::identity();
    let triangles = LoopSubdiv::create(&identity, &identity, reverse_orientation, params)?;
    let Some(CpuShape::Triangle(first_triangle)) = triangles.first() else {
        return Ok(None);
    };
    let mesh = &first_triangle.mesh;
    let positions = mesh
        .p
        .iter()
        .map(|p| Vec3f([p.x as f32, p.y as f32, p.z as f32]))
        .collect();
    let normals = if mesh.n.len() == mesh.p.len() {
        Some(
            mesh.n
                .iter()
                .map(|n| Vec3f([n.x as f32, n.y as f32, n.z as f32]))
                .collect(),
        )
    } else {
        None
    };
    let tangents = if mesh.s.len() == mesh.p.len() {
        Some(
            mesh.s
                .iter()
                .map(|s| Vec3f([s.x as f32, s.y as f32, s.z as f32]))
                .collect(),
        )
    } else {
        None
    };
    let uvs = if mesh.uv.len() == mesh.p.len() {
        Some(
            mesh.uv
                .iter()
                .map(|uv| Vec2f([uv.x as f32, uv.y as f32]))
                .collect(),
        )
    } else {
        None
    };
    Ok(Some(TriangleMeshShape {
        positions,
        indices: mesh.vertex_indices.clone(),
        normals,
        tangents,
        uvs,
    }))
}

/// Completes the attributes required by the flattened GPU mesh contract.
///
/// This intentionally runs before Flat IR construction.  In particular, a
/// mesh without normals is expanded to triangle corners so that a shared
/// vertex cannot accidentally turn flat shading into smooth shading.
pub fn complete_triangle_attributes(
    shape: TriangleMeshShape,
    node_name: &str,
) -> Result<TriangleMeshShape, PbrtError> {
    let vertex_count = shape.positions.len();
    if vertex_count == 0 || shape.indices.is_empty() || shape.indices.len() % 3 != 0 {
        return Err(PbrtError::error(&format!(
            "Shape node \"{}\" has no complete triangle mesh.",
            node_name
        )));
    }
    if !shape
        .positions
        .iter()
        .all(|position| position.0.iter().all(|value| value.is_finite()))
    {
        return Err(PbrtError::error(&format!(
            "Shape node \"{}\" contains a non-finite position.",
            node_name
        )));
    }
    for &index in &shape.indices {
        if index as usize >= vertex_count {
            return Err(PbrtError::error(&format!(
                "Shape node \"{}\" contains an out-of-range vertex index.",
                node_name
            )));
        }
    }
    validate_node_attribute_len(node_name, shape.normals.as_deref(), vertex_count, "normal")?;
    validate_node_attribute_len(
        node_name,
        shape.tangents.as_deref(),
        vertex_count,
        "tangent",
    )?;
    validate_node_attribute_len(node_name, shape.uvs.as_deref(), vertex_count, "UV")?;

    let uvs = shape
        .uvs
        .unwrap_or_else(|| generate_planar_uvs(&shape.positions));
    if !uvs
        .iter()
        .all(|uv| uv.0.iter().all(|value| value.is_finite()))
    {
        return Err(PbrtError::error(&format!(
            "Shape node \"{}\" contains a non-finite UV.",
            node_name
        )));
    }

    let source_tangents = shape.tangents;
    if let Some(tangents) = source_tangents.as_deref() {
        if !tangents.iter().all(|tangent| {
            tangent.0.iter().all(|value| value.is_finite()) && length_squared(tangent.0) > 0.0
        }) {
            return Err(PbrtError::error(&format!(
                "Shape node \"{}\" contains an invalid tangent.",
                node_name
            )));
        }
    }
    if shape.normals.is_none() {
        return expand_flat_mesh(
            shape.positions,
            shape.indices,
            uvs,
            source_tangents,
            node_name,
        );
    }

    let mut normals = shape.normals.unwrap();
    if !normals.iter().all(|normal| {
        normal.0.iter().all(|value| value.is_finite()) && length_squared(normal.0) > 0.0
    }) {
        return Err(PbrtError::error(&format!(
            "Shape node \"{}\" contains an invalid normal.",
            node_name
        )));
    }
    align_normals_to_winding(&shape.positions, &shape.indices, &mut normals);
    let tangents = source_tangents
        .unwrap_or_else(|| generate_tangents(&shape.positions, &shape.indices, &normals, &uvs));
    if !tangents.iter().all(|tangent| {
        tangent.0.iter().all(|value| value.is_finite()) && length_squared(tangent.0) > 0.0
    }) {
        return Err(PbrtError::error(&format!(
            "Shape node \"{}\" contains an invalid tangent.",
            node_name
        )));
    }
    Ok(TriangleMeshShape {
        positions: shape.positions,
        indices: shape.indices,
        normals: Some(normals),
        tangents: Some(tangents),
        uvs: Some(uvs),
    })
}

/// Flips per-vertex shading normals that disagree with the mesh winding.
///
/// Subdivision limit normals (e.g. loopsubdiv's `Cross(S, T)` tangents) can
/// wind opposite to the face orientation, which pbrt-v4 tolerates through its
/// hemisphere-agnostic BxDF conventions. The GPU backend additionally aligns
/// the shading normals with the geometric normals here so every downstream
/// stage can rely on a consistent orientation. Each vertex accumulates the
/// unnormalized (area-weighted) geometric normals of its incident triangles
/// and is flipped when it points against that average.
fn align_normals_to_winding(positions: &[Vec3f], indices: &[u32], normals: &mut [Vec3f]) {
    let mut accumulated = vec![[0.0f32; 3]; normals.len()];
    for triangle in indices.chunks_exact(3) {
        let p0 = positions[triangle[0] as usize].0;
        let p1 = positions[triangle[1] as usize].0;
        let p2 = positions[triangle[2] as usize].0;
        let face_normal = cross(sub(p1, p0), sub(p2, p0));
        if !face_normal.iter().all(|value| value.is_finite()) {
            continue;
        }
        for &corner in triangle {
            let entry = &mut accumulated[corner as usize];
            entry[0] += face_normal[0];
            entry[1] += face_normal[1];
            entry[2] += face_normal[2];
        }
    }
    for (normal, average) in normals.iter_mut().zip(&accumulated) {
        if dot(normal.0, *average) < 0.0 {
            normal.0 = [-normal.0[0], -normal.0[1], -normal.0[2]];
        }
    }
}

fn generate_planar_uvs(positions: &[Vec3f]) -> Vec<Vec2f> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for position in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position.0[axis]);
            max[axis] = max[axis].max(position.0[axis]);
        }
    }
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let drop_axis = if extent[0] <= extent[1] && extent[0] <= extent[2] {
        0
    } else if extent[1] <= extent[2] {
        1
    } else {
        2
    };
    let axes = match drop_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    positions
        .iter()
        .map(|position| {
            let coordinate = |axis: usize| {
                if extent[axis] > 0.0 {
                    (position.0[axis] - min[axis]) / extent[axis]
                } else {
                    0.0
                }
            };
            Vec2f([coordinate(axes[0]), coordinate(axes[1])])
        })
        .collect()
}

fn expand_flat_mesh(
    positions: Vec<Vec3f>,
    indices: Vec<u32>,
    uvs: Vec<Vec2f>,
    source_tangents: Option<Vec<Vec3f>>,
    node_name: &str,
) -> Result<TriangleMeshShape, PbrtError> {
    let mut expanded_positions = Vec::with_capacity(indices.len());
    let mut expanded_normals = Vec::with_capacity(indices.len());
    let mut expanded_tangents = Vec::with_capacity(indices.len());
    let mut expanded_uvs = Vec::with_capacity(indices.len());
    let mut expanded_indices = Vec::with_capacity(indices.len());
    for triangle in indices.chunks_exact(3) {
        let p = [
            positions[triangle[0] as usize],
            positions[triangle[1] as usize],
            positions[triangle[2] as usize],
        ];
        let uv = [
            uvs[triangle[0] as usize],
            uvs[triangle[1] as usize],
            uvs[triangle[2] as usize],
        ];
        let normal = cross(sub(p[1].0, p[0].0), sub(p[2].0, p[0].0));
        if !normal.iter().all(|value| value.is_finite()) || length_squared(normal) == 0.0 {
            return Err(PbrtError::error(&format!(
                "Shape node \"{}\" contains a zero-area triangle.",
                node_name
            )));
        }
        let normal = normalize(normal);
        let generated_tangent = triangle_tangent(p, uv, normal);
        for corner in 0..3 {
            expanded_positions.push(p[corner]);
            expanded_normals.push(Vec3f(normal));
            expanded_tangents.push(
                source_tangents
                    .as_ref()
                    .map(|tangents| tangents[triangle[corner] as usize])
                    .unwrap_or(Vec3f(generated_tangent)),
            );
            expanded_uvs.push(uv[corner]);
            expanded_indices.push((expanded_indices.len()) as u32);
        }
    }
    Ok(TriangleMeshShape {
        positions: expanded_positions,
        indices: expanded_indices,
        normals: Some(expanded_normals),
        tangents: Some(expanded_tangents),
        uvs: Some(expanded_uvs),
    })
}

fn generate_tangents(
    positions: &[Vec3f],
    indices: &[u32],
    normals: &[Vec3f],
    uvs: &[Vec2f],
) -> Vec<Vec3f> {
    let mut tangents = vec![[0.0; 3]; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let i = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let tangent = triangle_tangent(
            [positions[i[0]], positions[i[1]], positions[i[2]]],
            [uvs[i[0]], uvs[i[1]], uvs[i[2]]],
            normalize(normals[i[0]].0),
        );
        for index in i {
            tangents[index] = add(tangents[index], tangent);
        }
    }
    tangents
        .into_iter()
        .zip(normals)
        .map(|(tangent, normal)| {
            let tangent = sub(tangent, scale(normalize(normal.0), dot(tangent, normal.0)));
            Vec3f(if length_squared(tangent) > 0.0 {
                normalize(tangent)
            } else {
                coordinate_tangent(normalize(normal.0))
            })
        })
        .collect()
}

fn triangle_tangent(p: [Vec3f; 3], uv: [Vec2f; 3], normal: [f32; 3]) -> [f32; 3] {
    let e1 = sub(p[1].0, p[0].0);
    let e2 = sub(p[2].0, p[0].0);
    let du1 = uv[1].0[0] - uv[0].0[0];
    let dv1 = uv[1].0[1] - uv[0].0[1];
    let du2 = uv[2].0[0] - uv[0].0[0];
    let dv2 = uv[2].0[1] - uv[0].0[1];
    let determinant = du1 * dv2 - dv1 * du2;
    if determinant.abs() > 1e-9 {
        normalize(scale(
            sub(scale(e1, dv2), scale(e2, dv1)),
            1.0 / determinant,
        ))
    } else {
        coordinate_tangent(normal)
    }
}

fn coordinate_tangent(normal: [f32; 3]) -> [f32; 3] {
    if normal[0].abs() > 0.1 {
        normalize(cross([0.0, 1.0, 0.0], normal))
    } else {
        normalize(cross([1.0, 0.0, 0.0], normal))
    }
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length_squared(a: [f32; 3]) -> f32 {
    dot(a, a)
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let length = length_squared(a).sqrt();
    scale(a, 1.0 / length)
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

fn node_vec3_attribute(
    params: &ParameterDictionary,
    key: &str,
    vertex_count: usize,
) -> Result<Option<Vec<Vec3f>>, PbrtError> {
    let values = params.get_points(key);
    if values.is_empty() {
        if params.has_parameter(key) {
            return Err(PbrtError::error(&format!(
                "Mesh attribute \"{}\" must contain {} values.",
                key,
                vertex_count * 3
            )));
        }
        return Ok(None);
    }
    if values.len() != vertex_count * 3 {
        return Err(PbrtError::error(&format!(
            "Mesh attribute \"{}\" must contain {} values, got {}.",
            key,
            vertex_count * 3,
            values.len()
        )));
    }
    Ok(Some(
        values
            .chunks_exact(3)
            .map(|value| Vec3f([value[0] as f32, value[1] as f32, value[2] as f32]))
            .collect(),
    ))
}

fn node_vec2_attribute(
    params: &ParameterDictionary,
    key: &str,
    vertex_count: usize,
) -> Result<Option<Vec<Vec2f>>, PbrtError> {
    let values = params.get_points(key);
    if values.is_empty() {
        if params.has_parameter(key) {
            return Err(PbrtError::error(&format!(
                "Mesh attribute \"{}\" must contain {} values.",
                key,
                vertex_count * 2
            )));
        }
        return Ok(None);
    }
    if values.len() != vertex_count * 2 {
        return Err(PbrtError::error(&format!(
            "Mesh attribute \"{}\" must contain {} values, got {}.",
            key,
            vertex_count * 2,
            values.len()
        )));
    }
    Ok(Some(
        values
            .chunks_exact(2)
            .map(|value| Vec2f([value[0] as f32, value[1] as f32]))
            .collect(),
    ))
}

fn validate_node_attribute_len<T>(
    node_name: &str,
    attribute: Option<&[T]>,
    vertex_count: usize,
    attribute_name: &str,
) -> Result<(), PbrtError> {
    if let Some(attribute) = attribute {
        if attribute.len() != vertex_count {
            return Err(PbrtError::error(&format!(
                "Shape node \"{}\" has {} {} values for {} vertices.",
                node_name,
                attribute.len(),
                attribute_name,
                vertex_count
            )));
        }
    }
    Ok(())
}
