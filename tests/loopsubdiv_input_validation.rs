use pbrt_r4::base::shape::Shape;
use pbrt_r4::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

type Textures = HashMap<String, Arc<FloatTexture>>;

fn create(indices: &[i32], points: &[Float], levels: i32) -> Result<Vec<Shape>, PbrtError> {
    let mut params = ParameterDictionary::new();
    params.add_ints("indices", indices);
    params.add_point("P", points);
    params.add_int("levels", levels);
    Shape::create(
        "loopsubdiv",
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
        &Textures::new(),
    )
}

const TRIANGLE_P: [Float; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

fn boundary_mesh() -> (Vec<i32>, Vec<Float>) {
    (
        vec![0, 1, 2, 1, 3, 2],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    )
}

fn closed_mesh() -> (Vec<i32>, Vec<Float>) {
    (
        vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
}

#[test]
fn valid_meshes_still_build() {
    let shapes = create(&[0, 1, 2], &TRIANGLE_P, 0).unwrap();
    assert_eq!(shapes.len(), 1);

    for levels in [0, 1] {
        let (indices, points) = boundary_mesh();
        assert!(!create(&indices, &points, levels).unwrap().is_empty());
        let (indices, points) = closed_mesh();
        assert!(!create(&indices, &points, levels).unwrap().is_empty());
    }
}

#[test]
fn out_of_range_and_negative_indices_return_error() {
    assert!(create(&[0, 1, 9], &TRIANGLE_P, 0).is_err());
    assert!(create(&[0, -1, 2], &TRIANGLE_P, 0).is_err());
    assert!(create(&[0, 1, 3], &TRIANGLE_P, 0).is_err());
}

#[test]
fn partial_face_returns_error() {
    assert!(create(&[0, 1], &TRIANGLE_P, 0).is_err());
}
