use super::component::Component;
use super::node::Node;
use super::shape::{DiskShape, Shape, SphereShape, TriangleMeshShape};
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
            }
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
