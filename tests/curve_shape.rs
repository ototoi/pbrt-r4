use pbrt_r4::prelude::*;
use pbrt_r4::shapes::create_curve_shape;

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
