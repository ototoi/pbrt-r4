use pbrt_r4::util::base::{Point2f, Point3f, Vector3f};
use pbrt_r4::util::mesh::tri_quad_mesh::TriQuadMesh;

fn single_quad() -> TriQuadMesh {
    TriQuadMesh {
        p: vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(1.0, 1.0, 0.0),
        ],
        n: Vec::new(),
        uv: vec![
            Point2f::new(0.0, 0.0),
            Point2f::new(1.0, 0.0),
            Point2f::new(0.0, 1.0),
            Point2f::new(1.0, 1.0),
        ],
        face_indices: vec![7],
        tri_indices: Vec::new(),
        quad_indices: vec![0, 1, 2, 3],
    }
}

#[test]
fn convert_to_only_triangles_matches_v4_quad_order() {
    let mut mesh = single_quad();
    mesh.convert_to_only_triangles();
    assert_eq!(mesh.tri_indices, vec![0, 1, 3, 0, 3, 2]);
    assert!(mesh.quad_indices.is_empty());
    assert_eq!(mesh.face_indices, vec![7]);
}

#[test]
fn displacement_refines_displaces_and_recomputes_normals() {
    let mesh = single_quad();
    let mesh = mesh
        .displace(
            |a, b| Vector3f::distance(&a, &b),
            0.75,
            |p, n, uv| p + (uv.x + uv.y) * n,
        )
        .unwrap();

    assert!(mesh.quad_indices.is_empty());
    assert!(mesh.tri_indices.len() > 6);
    assert!(mesh.p.iter().any(|p| p.z > 0.0));
    assert_eq!(mesh.n.len(), mesh.p.len());
    assert!(mesh.n.iter().any(|n| n.z < 0.99));
}
