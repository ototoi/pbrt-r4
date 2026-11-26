#![cfg(feature = "mesh-accel")]

use pbrt_r4::base::shape::MeshGrouping;
use pbrt_r4::cpu::aggregates::bvh::build::SplitMethod;
use pbrt_r4::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

fn grouping() -> MeshGrouping {
    MeshGrouping {
        split_method: SplitMethod::HLBVH,
        max_prims_in_node: 8,
    }
}

fn base_params() -> ParameterDictionary {
    let mut params = ParameterDictionary::new();
    params.add_ints("indices", &[0, 1, 2, 0, 2, 3, 1, 4, 2, 1, 5, 4]);
    params.add_point(
        "P",
        &[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0,
            0.0,
        ],
    );
    params
}

fn create(name: &str, params: &ParameterDictionary) -> Vec<Shape> {
    Shape::create_meshes(
        name,
        &Transform::identity(),
        &Transform::identity(),
        false,
        params,
        &HashMap::<String, Arc<FloatTexture>>::new(),
        &grouping(),
    )
    .unwrap()
}

#[test]
fn trianglemesh_becomes_one_mesh_shape() {
    let shapes = create("trianglemesh", &base_params());
    assert_eq!(shapes.len(), 1);
    assert!(matches!(&shapes[0], Shape::Mesh(mesh) if mesh.triangle_count() == 4));
}

#[test]
fn alpha_trianglemesh_has_no_alpha_mask_shapes() {
    let mut params = base_params();
    params.add_float("alpha", 0.5);
    let shapes = create("trianglemesh", &params);
    assert_eq!(shapes.len(), 1);
    assert!(matches!(&shapes[0], Shape::Mesh(_)));
}

#[test]
fn plymesh_splits_triangles_and_quads() {
    let mut file = tempfile::Builder::new().suffix(".ply").tempfile().unwrap();
    use std::io::Write;
    write!(file, "ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nelement face 3\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n2 0 0\n2 1 0\n3 0 0\n3 1 0\n3 0 1 2\n3 0 2 3\n4 1 6 7 5\n").unwrap();
    let mut params = ParameterDictionary::new();
    params.add_string("filename", file.path().to_string_lossy().as_ref());
    let shapes = create("plymesh", &params);
    assert_eq!(shapes.len(), 2);
    assert!(shapes
        .iter()
        .any(|shape| matches!(shape, Shape::Mesh(mesh) if mesh.triangle_count() == 2)));
    assert!(shapes
        .iter()
        .any(|shape| matches!(shape, Shape::BilinearPatch(_))));
}

#[test]
fn loopsubdiv_becomes_one_mesh_shape() {
    let mut params = ParameterDictionary::new();
    params.add_int("levels", 1);
    params.add_ints("indices", &[0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2]);
    params.add_point(
        "P",
        &[
            1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
        ],
    );
    let expected = LoopSubdiv::create(
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
    )
    .unwrap()
    .len();
    let shapes = create("loopsubdiv", &params);
    assert_eq!(shapes.len(), 1);
    assert!(matches!(&shapes[0], Shape::Mesh(mesh) if mesh.triangle_count() == expected));
}

#[test]
fn degenerate_mesh_creates_no_mesh_shape() {
    let mut params = ParameterDictionary::new();
    params.add_ints("indices", &[0, 1, 2]);
    params.add_point("P", &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(
        create("trianglemesh", &params).len(),
        Shape::create(
            "trianglemesh",
            &Transform::identity(),
            &Transform::identity(),
            false,
            &params,
            &HashMap::new()
        )
        .unwrap()
        .len()
    );
}

#[test]
fn other_shapes_match_create() {
    let params = ParameterDictionary::new();
    let meshes = create("sphere", &params);
    let original = Shape::create(
        "sphere",
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(meshes.len(), original.len());
    assert!(matches!(meshes[0], Shape::Sphere(_)));
}
