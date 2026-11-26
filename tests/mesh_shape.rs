#![cfg(feature = "mesh-accel")]
use pbrt_r4::cpu::aggregates::bvh::build::SplitMethod;

use pbrt_r4::prelude::*;
use std::sync::Arc;

fn grid_mesh() -> (Arc<TriangleMesh>, Vec<u32>) {
    let side = 21;
    let mut points = Vec::with_capacity(side * side);
    let mut uv = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            let xf = x as Float / 20.0;
            let yf = y as Float / 20.0;
            points.push(Point3f::new(
                xf,
                yf,
                0.08 * (xf * 9.0).sin() * (yf * 7.0).cos(),
            ));
            uv.push(Point2f::new(xf, yf));
        }
    }
    let mut indices = Vec::with_capacity(800 * 3);
    for y in 0..20 {
        for x in 0..20 {
            let a = (y * side + x) as u32;
            let b = a + 1;
            let c = a + side as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    let faces = (0..(indices.len() / 3) as u32).collect();
    let mesh = Arc::new(TriangleMesh::new(
        &Transform::identity(),
        false,
        true,
        indices,
        points,
        vec![],
        vec![],
        uv,
    ));
    (mesh, faces)
}

fn ray(x: Float, y: Float) -> Ray {
    Ray::new(
        &Point3f::new(x, y, 2.0),
        &Vector3f::new(0.0, 0.0, -1.0),
        10.0,
        0.0,
    )
}

fn per_triangle(
    mesh: &Arc<TriangleMesh>,
    faces: &[u32],
    alpha: &Option<AlphaMaskInfo>,
    shadow: &Option<AlphaMaskInfo>,
) -> Vec<Shape> {
    faces
        .iter()
        .map(|face| {
            let triangle = Arc::new(Shape::Triangle(Triangle::new(mesh, *face)));
            if alpha.is_none() && shadow.is_none() {
                Arc::try_unwrap(triangle).ok().unwrap()
            } else {
                Shape::AlphaMask(Box::new(AlphaMaskShape::new(&triangle, alpha, shadow)))
            }
        })
        .collect()
}

fn nearest(shapes: &[Shape], ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
    let mut nearest = None;
    let mut limit = t_max;
    for shape in shapes {
        if let Some(hit) = shape.intersect(ray, limit) {
            limit = hit.t_hit;
            nearest = Some(hit);
        }
    }
    nearest
}

#[test]
fn mesh_hits_match_per_triangle() {
    let (mesh, faces) = grid_mesh();
    let mesh_shape =
        MeshShape::new(mesh.clone(), faces.clone(), None, SplitMethod::HLBVH, 8).unwrap();
    let shapes = per_triangle(&mesh, &faces, &None, &None);
    for j in 0..40 {
        for i in 0..40 {
            let r = ray((i as Float + 0.37) / 40.0, (j as Float + 0.61) / 40.0);
            let expected = nearest(&shapes, &r, 10.0);
            let actual = mesh_shape.intersect(&r, 10.0);
            match (expected, actual) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.t_hit, b.t_hit);
                    assert_eq!(a.intr.p, b.intr.p);
                    assert_eq!(a.intr.n, b.intr.n);
                    assert_eq!(a.intr.uv, b.intr.uv);
                }
                (None, None) => {}
                _ => panic!("mesh and per-triangle intersection differ"),
            }
            assert_eq!(
                mesh_shape.intersect_p(&r, 10.0),
                shapes.iter().any(|s| s.intersect_p(&r, 10.0))
            );
        }
    }
}

#[test]
fn mesh_respects_t_max_and_bounds() {
    let (mesh, faces) = grid_mesh();
    let mesh_shape = MeshShape::new(mesh, faces, None, SplitMethod::HLBVH, 8).unwrap();
    let outside = Ray::new(
        &Point3f::new(1.5, 1.5, 2.0),
        &Vector3f::new(0.0, 0.0, -1.0),
        10.0,
        0.0,
    );
    assert!(mesh_shape.intersect(&outside, 10.0).is_none());
    assert!(!mesh_shape.intersect_p(&outside, 10.0));
    let short = ray(0.4, 0.4);
    assert!(mesh_shape.intersect(&short, 1.0).is_none());
    assert!(!mesh_shape.intersect_p(&short, 1.0));
}

#[test]
fn mesh_alpha_matches_alpha_mask_shape() {
    let (mesh, faces) = grid_mesh();
    let alpha = Some(AlphaMaskInfo::Value { value: 0.5 });
    let shadow = Some(AlphaMaskInfo::Value { value: 0.5 });
    for (alpha, shadow) in [(alpha.clone(), None), (None, shadow.clone())] {
        let test = AlphaMaskTest::new(&alpha, &shadow);
        let mesh_shape = MeshShape::new(
            mesh.clone(),
            faces.clone(),
            Some(test),
            SplitMethod::HLBVH,
            8,
        )
        .unwrap();
        let shapes = per_triangle(&mesh, &faces, &alpha, &shadow);
        for j in 0..20 {
            for i in 0..20 {
                let r = ray((i as Float + 0.37) / 20.0, (j as Float + 0.61) / 20.0);
                let expected = nearest(&shapes, &r, 10.0);
                let actual = mesh_shape.intersect(&r, 10.0);
                assert_eq!(expected.map(|h| h.t_hit), actual.map(|h| h.t_hit));
                assert_eq!(
                    mesh_shape.intersect_p(&r, 10.0),
                    shapes.iter().any(|s| s.intersect_p(&r, 10.0))
                );
            }
        }
    }
}

#[test]
fn constant_zero_alpha_mesh_never_hits() {
    let (mesh, faces) = grid_mesh();
    let alpha = Some(AlphaMaskInfo::Value { value: 0.0 });
    let test = AlphaMaskTest::new(&alpha, &None);
    let mesh_shape = MeshShape::new(mesh, faces, Some(test), SplitMethod::HLBVH, 8).unwrap();
    let r = ray(0.4, 0.4);
    assert!(mesh_shape.intersect(&r, 10.0).is_none());
    assert!(!mesh_shape.intersect_p(&r, 10.0));
}
