use pbrt_r4::prelude::*;

use pbrt_r4::shapes::{create_curve_shape, create_curves_shape};

fn assert_close(a: Float, b: Float) {
    assert!((a - b).abs() < 1e-4, "{a} != {b}");
}

fn assert_vector_close(a: &Vector3f, b: &Vector3f) {
    assert_close(a.x, b.x);
    assert_close(a.y, b.y);
    assert_close(a.z, b.z);
}

fn curve_params(curve_type: &str) -> ParameterDictionary {
    let mut params = ParameterDictionary::new();
    params.add_string("type", curve_type);
    params.add_int("splitdepth", 0);
    params.add_point(
        "P",
        &[
            0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 0.0, 2.0,
        ],
    );
    params
}

fn intersectable_curve_params(curve_type: &str, split_depth: i32) -> ParameterDictionary {
    let mut params = ParameterDictionary::new();
    params.add_string("type", curve_type);
    params.add_int("splitdepth", split_depth);
    params.add_float("width", 0.2);
    params.add_point(
        "P",
        &[
            0.0, 0.0, 0.0, //
            0.3, 0.0, 0.0, //
            0.7, 0.0, 0.0, //
            1.0, 0.0, 0.0,
        ],
    );
    if curve_type == "ribbon" {
        params.add_point("N", &[0.0, -0.2, 0.98, 0.0, 0.2, 0.98]);
    }
    params
}

#[test]
fn create_curve_shape_rejects_unknown_basis() {
    let mut params = ParameterDictionary::new();
    params.add_string("basis", "invalid_basis");

    let result = create_curve_shape(
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
    );
    assert!(result.is_err());
}

#[test]
fn create_curve_shape_rejects_too_few_control_points() {
    let mut params = ParameterDictionary::new();
    params.add_int("degree", 3);
    params.add_string("basis", "bezier");
    params.add_point(
        "P",
        &[
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
        ],
    );

    let result = create_curve_shape(
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
    );
    assert!(result.is_err());
}

#[test]
fn create_curves_shape_preserves_input_shape_groups() {
    let transform = Transform::identity();
    let params = [curve_params("flat"), curve_params("flat")];

    let curve_sets = create_curves_shape(&transform, &transform, false, &params).unwrap();

    assert_eq!(curve_sets.len(), 2);
    assert_eq!(curve_sets[0].len(), 1);
    assert_eq!(curve_sets[1].len(), 1);
}

#[test]
fn shape_create_curves_accepts_multiple_parameter_dictionaries() {
    let transform = Transform::identity();
    let shape_params = [curve_params("flat"), curve_params("flat")];

    let shape_sets = Shape::create_curves(
        &transform,
        &transform,
        false,
        &shape_params,
        &Default::default(),
    )
    .unwrap();

    assert_eq!(shape_sets.len(), 2);
    assert!(shape_sets
        .iter()
        .all(|shapes| matches!(shapes.as_slice(), [Shape::Curve(_)])));
}

#[test]
fn create_curves_shape_rejects_mixed_curve_types() {
    let transform = Transform::identity();
    let params = [curve_params("flat"), curve_params("cylinder")];

    assert!(create_curves_shape(&transform, &transform, false, &params).is_err());
}

#[test]
fn grouped_curves_match_standalone_intersections_for_types_and_transforms() {
    let transforms = [
        Transform::identity(),
        Transform::translate(1.5, -0.5, 2.0),
        Transform::rotate_y(35.0),
        Transform::scale(1.5, 0.75, 2.0),
    ];

    for curve_type in ["flat", "ribbon", "cylinder"] {
        let params = intersectable_curve_params(curve_type, 0);
        for render_from_object in transforms {
            let object_from_render = render_from_object.inverse();
            let standalone =
                create_curve_shape(&render_from_object, &object_from_render, false, &params)
                    .unwrap();
            let grouped = Shape::create_curves(
                &render_from_object,
                &object_from_render,
                false,
                std::slice::from_ref(&params),
                &Default::default(),
            )
            .unwrap();

            let ray = Ray::new(
                &render_from_object.transform_point(&Point3f::new(0.5, 0.0, -1.0)),
                &render_from_object.transform_vector(&Vector3f::new(0.0, 0.0, 1.0)),
                Float::INFINITY,
                0.0,
            );
            let expected = standalone[0]
                .intersect(&ray, Float::INFINITY)
                .unwrap_or_else(|| panic!("standalone {curve_type} curve should intersect"));
            let actual = grouped[0][0]
                .intersect(&ray, Float::INFINITY)
                .unwrap_or_else(|| panic!("grouped {curve_type} curve should intersect"));

            assert_close(actual.t_hit, expected.t_hit);
            assert_vector_close(&actual.intr.p, &expected.intr.p);
            assert_vector_close(&actual.intr.n, &expected.intr.n);
            assert_vector_close(&actual.intr.wo, &expected.intr.wo);
        }
    }
}

#[test]
fn splitdepth_preserves_grouped_curve_intersections() {
    let transform = Transform::identity();
    let params = intersectable_curve_params("flat", 2);
    let standalone = create_curve_shape(&transform, &transform, false, &params).unwrap();
    let grouped =
        create_curves_shape(&transform, &transform, false, std::slice::from_ref(&params)).unwrap();

    assert_eq!(standalone.len(), 4);
    assert_eq!(grouped[0].len(), 4);
    let ray = Ray::new(
        &Point3f::new(0.125, 0.0, -1.0),
        &Vector3f::new(0.0, 0.0, 1.0),
        Float::INFINITY,
        0.0,
    );
    assert_eq!(
        standalone
            .iter()
            .any(|curve| curve.intersect_p(&ray, Float::INFINITY)),
        grouped[0]
            .iter()
            .any(|curve| curve.intersect_p(&ray, Float::INFINITY))
    );
}
