#![cfg(feature = "mesh-accel")]

use pbrt_r4::cpu::aggregates::bvh::build::SplitMethod;
use pbrt_r4::prelude::*;
use std::sync::Arc;

fn grid_mesh() -> (Arc<TriangleMesh>, Vec<u32>) {
    let side = 21;
    let layer_size = side * side;
    let mut points = Vec::with_capacity(layer_size * 2);
    let mut uv = Vec::with_capacity(layer_size * 2);
    for layer in 0..2 {
        for y in 0..side {
            for x in 0..side {
                let xf = x as Float / 20.0;
                let yf = y as Float / 20.0;
                let z =
                    0.08 * (xf * 9.0).sin() * (yf * 7.0).cos() + if layer == 0 { 0.0 } else { 0.5 };
                points.push(Point3f::new(xf, yf, z));
                uv.push(Point2f::new(xf + layer as Float, yf));
            }
        }
    }
    let mut indices = Vec::with_capacity(1600 * 3);
    for layer in 0..2 {
        let base = (layer * layer_size) as u32;
        for y in 0..20 {
            for x in 0..20 {
                let a = base + (y * side + x) as u32;
                let b = a + 1;
                let c = a + side as u32;
                let d = c + 1;
                indices.extend_from_slice(&[a, b, d, a, d, c]);
            }
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

fn rays() -> Vec<Ray> {
    let mut state = 0x5eed_u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as Float / u32::MAX as Float
    };
    (0..2000)
        .map(|_| {
            let x = -0.1 + next() * 1.2;
            let y = -0.1 + next() * 1.2;
            let dx = -0.3 + next() * 0.6;
            let dy = -0.3 + next() * 0.6;
            Ray::new(
                &Point3f::new(x, y, 1.5),
                &Vector3f::new(dx, dy, -1.0),
                10.0,
                0.0,
            )
        })
        .collect()
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

fn intersections(shapes: &[Shape], ray: &Ray, t_max: Float) -> Vec<ShapeIntersection> {
    let mut hits: Vec<_> = shapes
        .iter()
        .filter_map(|shape| shape.intersect(ray, t_max))
        .collect();
    hits.sort_by(|a, b| a.t_hit.total_cmp(&b.t_hit));
    hits
}

fn assert_matches(mesh_shape: &MeshShape, shapes: &[Shape], rays: &[Ray]) {
    for ray in rays {
        let hits = intersections(shapes, ray, 10.0);
        if hits.len() > 1 && hits[0].t_hit == hits[1].t_hit {
            continue;
        }
        let expected = hits.first();
        let actual = mesh_shape.intersect(ray, 10.0);
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
            mesh_shape.intersect_p(ray, 10.0),
            shapes.iter().any(|shape| shape.intersect_p(ray, 10.0))
        );
    }
}

fn checkerboard_alpha() -> AlphaMaskInfo {
    let one = Arc::new(FloatTexture::Constant(ConstantTexture::new(&1.0)));
    let zero = Arc::new(FloatTexture::Constant(ConstantTexture::new(&0.0)));
    let checker = FloatTexture::Checkerboard2D(Checkerboard2DTexture::new(
        TextureMapping2D::UV(UVMapping::new(1.0, 1.0, 0.0, 0.0)),
        &zero,
        &one,
        AAMethod::None,
    ));
    AlphaMaskInfo::Texture {
        texture: Arc::new(checker),
    }
}

#[test]
fn mesh_hits_match_per_triangle_for_random_oblique_rays() {
    let (mesh, faces) = grid_mesh();
    let mesh_shape =
        MeshShape::new(mesh.clone(), faces.clone(), None, SplitMethod::HLBVH, 8).unwrap();
    let shapes = per_triangle(&mesh, &faces, &None, &None);
    assert_matches(&mesh_shape, &shapes, &rays());
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
    let short = Ray::new(
        &Point3f::new(0.4, 0.4, 1.5),
        &Vector3f::new(0.0, 0.0, -1.0),
        10.0,
        0.0,
    );
    assert!(mesh_shape.intersect(&short, 0.4).is_none());
    assert!(!mesh_shape.intersect_p(&short, 0.4));
}

#[test]
fn mesh_alpha_and_shadow_masks_match_alpha_mask_shape() {
    let (mesh, faces) = grid_mesh();
    let rays = rays();
    let constant_half = Some(AlphaMaskInfo::Value { value: 0.5 });
    let checker = Some(checkerboard_alpha());
    let cases = [
        (constant_half.clone(), None),
        (None, constant_half.clone()),
        (checker, None),
    ];
    for (alpha, shadow) in cases {
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
        assert_matches(&mesh_shape, &shapes, &rays);
    }
}

#[test]
fn constant_zero_alpha_mesh_never_hits() {
    let (mesh, faces) = grid_mesh();
    let alpha = Some(AlphaMaskInfo::Value { value: 0.0 });
    let test = AlphaMaskTest::new(&alpha, &None);
    let mesh_shape = MeshShape::new(mesh, faces, Some(test), SplitMethod::HLBVH, 8).unwrap();
    let r = Ray::new(
        &Point3f::new(0.4, 0.4, 1.5),
        &Vector3f::new(0.0, 0.0, -1.0),
        10.0,
        0.0,
    );
    assert!(mesh_shape.intersect(&r, 10.0).is_none());
    assert!(!mesh_shape.intersect_p(&r, 10.0));
}

#[test]
fn uv_texture_alpha_passes_through_to_the_farther_layer() {
    let (mesh, faces) = grid_mesh();
    let alpha = Some(checkerboard_alpha());
    let mesh_shape = MeshShape::new(
        mesh,
        faces,
        Some(AlphaMaskTest::new(&alpha, &None)),
        SplitMethod::HLBVH,
        8,
    )
    .unwrap();
    let ray = Ray::new(
        &Point3f::new(0.4, 0.4, 1.5),
        &Vector3f::new(0.0, 0.0, -1.0),
        10.0,
        0.0,
    );
    let hit = mesh_shape
        .intersect(&ray, 10.0)
        .expect("farther layer should pass alpha");
    assert!(
        hit.t_hit < 1.1,
        "expected the lower layer, got t={}",
        hit.t_hit
    );
}
