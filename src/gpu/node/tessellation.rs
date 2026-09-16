use super::component::Component;
use super::node::Node;
use super::shape::{
    BilinearMeshShape, ConeShape, CylinderShape, DiskShape, HeightFieldShape, HyperboloidShape,
    NurbsShape, ParaboloidShape, Shape, SphereShape, TriangleMeshShape,
};
use super::types::{Vec2f, Vec3f};
use crate::util::error::PbrtError;

pub const DEFAULT_SPHERE_PHI_SEGMENTS: usize = 32;
pub const DEFAULT_SPHERE_THETA_SEGMENTS: usize = 16;

pub fn tessellate_shapes(node: &mut Node) -> Result<(), PbrtError> {
    for component in &mut node.components {
        if let Component::Shape(shape_component) = component {
            if let Shape::Sphere(sphere) = &shape_component.shape {
                shape_component.shape = Shape::TriangleMesh(Box::new(sphere_to_mesh(sphere)));
            } else if let Shape::Disk(disk) = &shape_component.shape {
                shape_component.shape = Shape::TriangleMesh(Box::new(disk_to_mesh(disk)?));
            } else if let Shape::Cylinder(cylinder) = &shape_component.shape {
                shape_component.shape = Shape::TriangleMesh(Box::new(cylinder_to_mesh(cylinder)?));
            } else if let Shape::Cone(cone) = &shape_component.shape {
                shape_component.shape = Shape::TriangleMesh(Box::new(cone_to_mesh(cone)?));
            } else if let Shape::Paraboloid(shape) = &shape_component.shape {
                shape_component.shape = Shape::TriangleMesh(Box::new(paraboloid_to_mesh(shape)?));
            } else if let Shape::HeightField(shape) = &shape_component.shape {
                shape_component.shape = Shape::TriangleMesh(Box::new(heightfield_to_mesh(shape)?));
            } else if let Shape::BilinearMesh(shape) = &shape_component.shape {
                shape_component.shape = Shape::TriangleMesh(Box::new(bilinear_to_mesh(shape)?));
            } else if let Shape::Hyperboloid(shape) = &shape_component.shape {
                shape_component.shape = Shape::TriangleMesh(Box::new(hyperboloid_to_mesh(shape)?));
            } else if let Shape::Nurbs(shape) = &shape_component.shape {
                shape_component.shape = Shape::TriangleMesh(Box::new(nurbs_to_mesh(shape)?));
            }
        }
    }
    for component in &node.components {
        if let Component::Instance(instance) = component {
            let mut target = instance
                .instance
                .target
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            tessellate_shapes(&mut target)?;
        }
    }
    for child in &node.children {
        let mut child = child
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        tessellate_shapes(&mut child)?;
    }
    Ok(())
}

fn paraboloid_to_mesh(shape: &ParaboloidShape) -> Result<TriangleMeshShape, PbrtError> {
    let p = &shape.params;
    let radius = p.get_one_float("radius", 1.0);
    let mut zmin = p.get_one_float("zmin", 0.0);
    let mut zmax = p.get_one_float("zmax", 1.0);
    if zmin > zmax {
        std::mem::swap(&mut zmin, &mut zmax);
    }
    let phimax = p.get_one_float("phimax", 360.0).clamp(0.0, 360.0);
    if ![radius, zmin, zmax, phimax].iter().all(|v| v.is_finite())
        || radius <= 0.0
        || zmax <= zmin
        || phimax <= 0.0
    {
        return Err(PbrtError::error(
            "Paraboloid has invalid tessellation parameters.",
        ));
    }
    let us = p.get_one_int("udiv", 64).max(4) as usize;
    let vs = p.get_one_int("vdiv", 16).max(1) as usize;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut tangents = Vec::new();
    let mut uvs = Vec::new();
    for j in 0..=vs {
        let v = j as f32 / vs as f32;
        let z = zmin + (zmax - zmin) * v;
        let r = radius * ((z / zmax).max(0.0)).sqrt();
        for i in 0..=us {
            let u = i as f32 / us as f32;
            let phi = (phimax as f32).to_radians() * u;
            let (s, c) = phi.sin_cos();
            positions.push(Vec3f([r as f32 * c, r as f32 * s, z as f32]));
            let nx = r as f32 * c;
            let ny = r as f32 * s;
            let nz = -(radius * radius / zmax) as f32;
            let inv = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt().max(f32::MIN_POSITIVE);
            normals.push(Vec3f([nx * inv, ny * inv, nz * inv]));
            tangents.push(Vec3f([-s, c, 0.0]));
            uvs.push(Vec2f([u, v]));
        }
    }
    let mut indices = Vec::new();
    for j in 0..vs {
        for i in 0..us {
            let a = (j * (us + 1) + i) as u32;
            let c = a + (us + 1) as u32;
            indices.extend_from_slice(&[a, a + 1, c + 1, a, c + 1, c]);
        }
    }
    Ok(TriangleMeshShape {
        positions,
        indices,
        normals: Some(normals),
        tangents: Some(tangents),
        uvs: Some(uvs),
    })
}

fn heightfield_to_mesh(shape: &HeightFieldShape) -> Result<TriangleMeshShape, PbrtError> {
    let p = &shape.params;
    let nx = p.get_one_int("nu", -1);
    let ny = p.get_one_int("nv", -1);
    let z = p
        .get_floats_ref("Pz")
        .ok_or_else(|| PbrtError::error("HeightField requires Pz."))?;
    if nx < 2 || ny < 2 || z.len() != nx as usize * ny as usize {
        return Err(PbrtError::error(
            "HeightField has invalid resolution or Pz count.",
        ));
    }
    let (nx, ny) = (nx as usize, ny as usize);
    let mut positions = Vec::with_capacity(nx * ny);
    let mut uvs = Vec::with_capacity(nx * ny);
    for y in 0..ny {
        for x in 0..nx {
            let u = x as f32 / (nx - 1) as f32;
            let v = y as f32 / (ny - 1) as f32;
            positions.push(Vec3f([u, v, z[y * nx + x] as f32]));
            uvs.push(Vec2f([u, v]));
        }
    }
    let mut indices = Vec::with_capacity((nx - 1) * (ny - 1) * 6);
    for y in 0..ny - 1 {
        for x in 0..nx - 1 {
            let a = (y * nx + x) as u32;
            let b = a + 1;
            let d = a + nx as u32;
            let c = d + 1;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    Ok(TriangleMeshShape {
        positions,
        indices,
        normals: None,
        tangents: None,
        uvs: Some(uvs),
    })
}

fn bilinear_to_mesh(shape: &BilinearMeshShape) -> Result<TriangleMeshShape, PbrtError> {
    let p = &shape.params;
    let raw = p.get_points("P");
    if raw.len() < 12 || raw.len() % 12 != 0 {
        return Err(PbrtError::error(
            "BilinearMesh requires groups of four points.",
        ));
    }
    let positions = raw
        .chunks_exact(3)
        .map(|v| Vec3f([v[0] as f32, v[1] as f32, v[2] as f32]))
        .collect::<Vec<_>>();
    let normals = p.get_points("N").into_iter().collect::<Vec<_>>();
    let normals = if normals.is_empty() {
        None
    } else if normals.len() == positions.len() * 3 {
        Some(
            normals
                .chunks_exact(3)
                .map(|v| Vec3f([v[0] as f32, v[1] as f32, v[2] as f32]))
                .collect(),
        )
    } else {
        return Err(PbrtError::error(
            "BilinearMesh normal count does not match P.",
        ));
    };
    let uv_values = p.get_floats("uv");
    let uvs = if uv_values.is_empty() {
        None
    } else if uv_values.len() == positions.len() * 2 {
        Some(
            uv_values
                .chunks_exact(2)
                .map(|v| Vec2f([v[0] as f32, v[1] as f32]))
                .collect(),
        )
    } else {
        return Err(PbrtError::error("BilinearMesh UV count does not match P."));
    };
    let mut indices = Vec::new();
    let raw_indices = p.get_ints("indices");
    let quads: Vec<u32> = if raw_indices.is_empty() {
        (0..positions.len() as u32).collect()
    } else {
        if raw_indices.len() % 4 != 0 {
            return Err(PbrtError::error(
                "BilinearMesh indices must contain groups of four.",
            ));
        }
        raw_indices.into_iter().map(|i| i as u32).collect()
    };
    for quad in quads.chunks_exact(4) {
        let [a, b, c, d] = [quad[0], quad[1], quad[2], quad[3]];
        if [a, b, c, d].iter().any(|&i| i as usize >= positions.len()) {
            return Err(PbrtError::error("BilinearMesh index is out of range."));
        }
        indices.extend_from_slice(&[a, b, d, a, d, c]);
    }
    Ok(TriangleMeshShape {
        positions,
        indices,
        normals,
        tangents: None,
        uvs,
    })
}

fn hyperboloid_to_mesh(shape: &HyperboloidShape) -> Result<TriangleMeshShape, PbrtError> {
    let p = &shape.params;
    let a = p.get_points("p1");
    let b = p.get_points("p2");
    if a.len() != 3 || b.len() != 3 {
        return Err(PbrtError::error("Hyperboloid requires p1 and p2."));
    }
    let zmin = a[2].min(b[2]);
    let zmax = a[2].max(b[2]);
    let radius = ((a[0] * a[0] + a[1] * a[1]).max(b[0] * b[0] + b[1] * b[1])).sqrt();
    let phi = p.get_one_float("phimax", 360.0).clamp(0.0, 360.0);
    if ![zmin, zmax, radius, phi].iter().all(|v| v.is_finite())
        || zmax <= zmin
        || radius <= 0.0
        || phi <= 0.0
    {
        return Err(PbrtError::error(
            "Hyperboloid has invalid tessellation parameters.",
        ));
    }
    let us = p.get_one_int("udiv", 64).max(4) as usize;
    let vs = p.get_one_int("vdiv", 16).max(1) as usize;
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    for j in 0..=vs {
        let v = j as f32 / vs as f32;
        let z = zmin + (zmax - zmin) * v;
        let base_x = a[0] + (b[0] - a[0]) * v;
        let base_y = a[1] + (b[1] - a[1]) * v;
        let row_radius = (base_x * base_x + base_y * base_y).sqrt() as f32;
        for i in 0..=us {
            let u = i as f32 / us as f32;
            let q = (phi as f32).to_radians() * u;
            let (s, c) = q.sin_cos();
            positions.push(Vec3f([row_radius * c, row_radius * s, z as f32]));
            uvs.push(Vec2f([u, v]));
        }
    }
    let mut indices = Vec::new();
    for j in 0..vs {
        for i in 0..us {
            let x = (j * (us + 1) + i) as u32;
            let y = x + (us + 1) as u32;
            indices.extend_from_slice(&[x, x + 1, y + 1, x, y + 1, y]);
        }
    }
    Ok(TriangleMeshShape {
        positions,
        indices,
        normals: None,
        tangents: None,
        uvs: Some(uvs),
    })
}

fn nurbs_to_mesh(shape: &NurbsShape) -> Result<TriangleMeshShape, PbrtError> {
    let identity = crate::util::transform::Transform::identity();
    let shapes = crate::shapes::nurbs::create_nurbs(&identity, &identity, false, &shape.params)?;
    let Some(crate::base::shape::Shape::Triangle(first)) = shapes.first() else {
        return Err(PbrtError::error("NURBS produced no triangles."));
    };
    let mesh = &first.mesh;
    let positions = mesh
        .p
        .iter()
        .map(|p| Vec3f([p.x as f32, p.y as f32, p.z as f32]))
        .collect();
    let indices = mesh.vertex_indices.clone();
    let normals = Some(
        mesh.n
            .iter()
            .map(|n| Vec3f([n.x as f32, n.y as f32, n.z as f32]))
            .collect(),
    );
    let tangents = (mesh.s.len() == mesh.p.len()).then(|| {
        mesh.s
            .iter()
            .map(|s| Vec3f([s.x as f32, s.y as f32, s.z as f32]))
            .collect()
    });
    let uvs = Some(
        mesh.uv
            .iter()
            .map(|uv| Vec2f([uv.x as f32, uv.y as f32]))
            .collect(),
    );
    Ok(TriangleMeshShape {
        positions,
        indices,
        normals,
        tangents,
        uvs,
    })
}

fn cylinder_to_mesh(cylinder: &CylinderShape) -> Result<TriangleMeshShape, PbrtError> {
    let params = &cylinder.params;
    let radius = params.get_one_float("radius", 1.0);
    let mut zmin = params.get_one_float("zmin", -1.0);
    let mut zmax = params.get_one_float("zmax", 1.0);
    if zmin > zmax {
        std::mem::swap(&mut zmin, &mut zmax);
    }
    let phimax = params.get_one_float("phimax", 360.0).clamp(0.0, 360.0);
    if ![radius, zmin, zmax, phimax].iter().all(|v| v.is_finite())
        || radius <= 0.0
        || zmin >= zmax
        || phimax <= 0.0
    {
        return Err(PbrtError::error(
            "Cylinder has invalid tessellation parameters.",
        ));
    }
    let segments = params.get_one_int("udiv", 64).max(4) as usize;
    let mut positions = Vec::with_capacity((segments + 1) * 2);
    let mut normals = Vec::with_capacity((segments + 1) * 2);
    let mut tangents = Vec::with_capacity((segments + 1) * 2);
    let mut uvs = Vec::with_capacity((segments + 1) * 2);
    for j in 0..=segments {
        let u = j as f32 / segments as f32;
        let phi = (phimax as f32).to_radians() * u;
        let (s, c) = phi.sin_cos();
        for (v, z) in [(0.0, zmin), (1.0, zmax)] {
            positions.push(Vec3f([radius as f32 * c, radius as f32 * s, z as f32]));
            normals.push(Vec3f([c, s, 0.0]));
            tangents.push(Vec3f([-s, c, 0.0]));
            uvs.push(Vec2f([u, v]));
        }
    }
    let mut indices = Vec::with_capacity(segments * 6);
    for j in 0..segments {
        let i = (j * 2) as u32;
        indices.extend_from_slice(&[i, i + 3, i + 1, i, i + 2, i + 3]);
    }
    Ok(TriangleMeshShape {
        positions,
        indices,
        normals: Some(normals),
        tangents: Some(tangents),
        uvs: Some(uvs),
    })
}

fn cone_to_mesh(cone: &ConeShape) -> Result<TriangleMeshShape, PbrtError> {
    let params = &cone.params;
    let radius = params.get_one_float("radius", 1.0);
    let height = params.get_one_float("height", 1.0);
    let phimax = params.get_one_float("phimax", 360.0).clamp(0.0, 360.0);
    if ![radius, height, phimax].iter().all(|v| v.is_finite())
        || radius <= 0.0
        || height == 0.0
        || phimax <= 0.0
    {
        return Err(PbrtError::error(
            "Cone has invalid tessellation parameters.",
        ));
    }
    let segments = params.get_one_int("udiv", 64).max(4) as usize;
    let mut positions = Vec::with_capacity((segments + 1) * 2);
    let mut normals = Vec::with_capacity((segments + 1) * 2);
    let mut tangents = Vec::with_capacity((segments + 1) * 2);
    let mut uvs = Vec::with_capacity((segments + 1) * 2);
    let slope = radius / height;
    for j in 0..=segments {
        let u = j as f32 / segments as f32;
        let phi = (phimax as f32).to_radians() * u;
        let (s, c) = phi.sin_cos();
        for (v, z) in [(0.0, 0.0), (1.0, height)] {
            let r = radius * (1.0 - v);
            positions.push(Vec3f([r as f32 * c, r as f32 * s, z as f32]));
            let inv_len = (1.0 / (1.0 + slope * slope).sqrt()) as f32;
            let n = Vec3f([c * inv_len, s * inv_len, slope as f32 * inv_len]);
            normals.push(n);
            tangents.push(Vec3f([-s, c, 0.0]));
            uvs.push(Vec2f([u, v]));
        }
    }
    let mut indices = Vec::with_capacity(segments * 6);
    for j in 0..segments {
        let i = (j * 2) as u32;
        indices.extend_from_slice(&[i, i + 3, i + 1, i, i + 2, i + 3]);
    }
    Ok(TriangleMeshShape {
        positions,
        indices,
        normals: Some(normals),
        tangents: Some(tangents),
        uvs: Some(uvs),
    })
}

pub const DEFAULT_DISK_PHI_SEGMENTS: usize = 64;
pub const DEFAULT_DISK_RADIAL_SEGMENTS: usize = 1;

fn disk_to_mesh(disk: &DiskShape) -> Result<TriangleMeshShape, PbrtError> {
    let params = &disk.params;
    let height = params.get_one_float("height", 0.0);
    let radius = params.get_one_float("radius", 1.0);
    let inner_radius = params.get_one_float("innerradius", 0.0);
    let phi_input = params.get_one_float("phimax", 360.0);
    let phi_degrees = phi_input.clamp(0.0, 360.0);
    let phi_max = phi_degrees.to_radians();
    let phi_segments = params
        .get_one_int("udiv", DEFAULT_DISK_PHI_SEGMENTS as i32)
        .max(4) as usize;
    let radial_segments = params
        .get_one_int("vdiv", DEFAULT_DISK_RADIAL_SEGMENTS as i32)
        .max(1) as usize;

    if ![height, radius, inner_radius, phi_input]
        .iter()
        .all(|value| value.is_finite())
        || radius <= 0.0
        || inner_radius < 0.0
        || inner_radius >= radius
        || phi_max <= 0.0
    {
        return Err(PbrtError::error(
            "Disk has invalid tessellation parameters.",
        ));
    }

    let ring_count = if inner_radius == 0.0 {
        radial_segments
    } else {
        radial_segments + 1
    };
    let mut positions = Vec::with_capacity(ring_count * (phi_segments + 1) + 1);
    let mut normals = Vec::with_capacity(positions.capacity());
    let mut tangents = Vec::with_capacity(positions.capacity());
    let mut uvs = Vec::with_capacity(positions.capacity());

    let add_vertex = |position: Vec3f,
                      u: f32,
                      v: f32,
                      tangent: Vec3f,
                      positions: &mut Vec<Vec3f>,
                      normals: &mut Vec<Vec3f>,
                      tangents: &mut Vec<Vec3f>,
                      uvs: &mut Vec<Vec2f>| {
        positions.push(position);
        normals.push(Vec3f([0.0, 0.0, 1.0]));
        tangents.push(tangent);
        uvs.push(Vec2f([u, v]));
    };

    let first_ring = if inner_radius == 0.0 { 1 } else { 0 };
    for ring in first_ring..=radial_segments {
        let radius_t = ring as f32 / radial_segments as f32;
        let ring_radius = inner_radius + (radius - inner_radius) * radius_t;
        for segment in 0..=phi_segments {
            let u = segment as f32 / phi_segments as f32;
            let (sin_phi, cos_phi) = (phi_max * u).sin_cos();
            let tangent = Vec3f([-sin_phi, cos_phi, 0.0]);
            add_vertex(
                Vec3f([ring_radius * cos_phi, ring_radius * sin_phi, height]),
                u,
                1.0 - radius_t,
                tangent,
                &mut positions,
                &mut normals,
                &mut tangents,
                &mut uvs,
            );
        }
    }

    let mut indices = Vec::new();
    if inner_radius == 0.0 {
        let center = positions.len() as u32;
        add_vertex(
            Vec3f([0.0, 0.0, height]),
            0.0,
            1.0,
            Vec3f([1.0, 0.0, 0.0]),
            &mut positions,
            &mut normals,
            &mut tangents,
            &mut uvs,
        );
        tangents[center as usize] = Vec3f([1.0, 0.0, 0.0]);
        let outer_start = 0u32;
        for segment in 0..phi_segments {
            indices.extend_from_slice(&[
                center,
                outer_start + segment as u32,
                outer_start + segment as u32 + 1,
            ]);
        }
        for ring in 1..radial_segments {
            append_disk_ring_quads(&mut indices, ring - 1, ring, phi_segments);
        }
    } else {
        for ring in 0..radial_segments {
            append_disk_ring_quads(&mut indices, ring, ring + 1, phi_segments);
        }
    }

    Ok(TriangleMeshShape {
        positions,
        indices,
        normals: Some(normals),
        tangents: Some(tangents),
        uvs: Some(uvs),
    })
}

fn append_disk_ring_quads(
    indices: &mut Vec<u32>,
    inner_ring: usize,
    outer_ring: usize,
    phi_segments: usize,
) {
    let inner_start = (inner_ring * (phi_segments + 1)) as u32;
    let outer_start = (outer_ring * (phi_segments + 1)) as u32;
    for segment in 0..phi_segments {
        let i0 = inner_start + segment as u32;
        let i1 = i0 + 1;
        let o0 = outer_start + segment as u32;
        let o1 = o0 + 1;
        indices.extend_from_slice(&[i0, o0, i1, i1, o0, o1]);
    }
}

fn sphere_to_mesh(sphere: &SphereShape) -> TriangleMeshShape {
    let phi_segments = sphere
        .params
        .get_one_int("udiv", DEFAULT_SPHERE_PHI_SEGMENTS as i32)
        .max(3) as usize;
    let theta_segments = sphere
        .params
        .get_one_int("vdiv", DEFAULT_SPHERE_THETA_SEGMENTS as i32)
        .max(2) as usize;
    let radius = sphere.params.get_one_float("radius", 1.0);
    let z_min = sphere.params.get_one_float("zmin", -radius);
    let z_max = sphere.params.get_one_float("zmax", radius);
    let phi_max = sphere.params.get_one_float("phimax", 360.0).to_radians();
    let mut positions = Vec::with_capacity((phi_segments + 1) * (theta_segments + 1));
    let mut normals = Vec::with_capacity(positions.capacity());
    let mut tangents = Vec::with_capacity(positions.capacity());
    let mut uvs = Vec::with_capacity(positions.capacity());
    let theta_min = (z_max / radius).clamp(-1.0, 1.0).acos();
    let theta_max = (z_min / radius).clamp(-1.0, 1.0).acos();
    for y in 0..=theta_segments {
        let v = y as f32 / theta_segments as f32;
        let theta = theta_min + (theta_max - theta_min) * v;
        let z = radius * theta.cos();
        let r = radius * theta.sin();
        for x in 0..=phi_segments {
            let u = x as f32 / phi_segments as f32;
            let (sin_phi, cos_phi) = (phi_max * u).sin_cos();
            let position = Vec3f([r * cos_phi, r * sin_phi, z]);
            positions.push(position);
            normals.push(Vec3f([
                position.0[0] / radius,
                position.0[1] / radius,
                position.0[2] / radius,
            ]));
            // Use the normalized direction of dpdu. At the poles dpdu has
            // zero length, but this longitude-dependent limit remains a
            // finite, non-zero tangent for every duplicated pole vertex.
            tangents.push(Vec3f([-sin_phi, cos_phi, 0.0]));
            uvs.push(Vec2f([u, v]));
        }
    }
    let mut indices = Vec::with_capacity(theta_segments * phi_segments * 6);
    for y in 0..theta_segments {
        for x in 0..phi_segments {
            let a = (y * (phi_segments + 1) + x) as u32;
            let b = a + 1;
            let c = a + (phi_segments + 1) as u32;
            let d = c + 1;
            if y == 0 {
                // All vertices in the first row are the north pole. Use one
                // of them as the fan center instead of connecting two pole
                // vertices, which would produce a zero-area triangle.
                indices.extend_from_slice(&[a, c, d]);
            } else if y + 1 == theta_segments {
                // The last row is the south pole. As above, emit only the
                // non-degenerate triangle for this side of the fan.
                indices.extend_from_slice(&[a, c, b]);
            } else {
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
    }
    TriangleMeshShape {
        positions,
        indices,
        normals: Some(normals),
        tangents: Some(tangents),
        uvs: Some(uvs),
    }
}
