use super::component::Component;
use super::node::Node;
use super::shape::{Shape, SphereShape, TriangleMeshShape};
use super::types::{Vec2f, Vec3f};

pub const DEFAULT_SPHERE_PHI_SEGMENTS: usize = 32;
pub const DEFAULT_SPHERE_THETA_SEGMENTS: usize = 16;

pub fn tessellate_shapes(node: &mut Node) {
    for component in &mut node.components {
        if let Component::Shape(shape_component) = component {
            if let Shape::Sphere(sphere) = &shape_component.shape {
                shape_component.shape = Shape::TriangleMesh(Box::new(sphere_to_mesh(sphere)));
            }
        }
    }
    for child in &node.children {
        let mut child = child
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        tessellate_shapes(&mut child);
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
            indices.extend_from_slice(&[a, c, b, b, c, d]);
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
