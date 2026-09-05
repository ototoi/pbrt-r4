use pbrt_r4::gpu::ir::flat::{build_light_bounds, Bounds3, LightBoundInput, LightBounds};

#[test]
fn point_light_bounds_match_v4_shape() {
    let bounds = build_light_bounds(&[LightBoundInput::Point {
        handle: 0,
        world_position: [1.0, 2.0, 3.0],
        intensity_max: 2.0,
        scale: 0.5,
    }])
    .unwrap();
    let point = bounds[0];
    assert_eq!(point.bounds.min, [1.0, 2.0, 3.0]);
    assert_eq!(point.bounds.max, [1.0, 2.0, 3.0]);
    assert_eq!(point.direction, [0.0, 0.0, 1.0]);
    assert_eq!(point.phi, 4.0 * std::f32::consts::PI);
    assert_eq!(point.cos_theta_o, -1.0);
    assert_eq!(point.cos_theta_e, 0.0);
}

#[test]
fn area_triangle_bounds_apply_orientation_and_area() {
    let bounds = build_light_bounds(&[LightBoundInput::AreaTriangle {
        handle: 0,
        world_positions: [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        input_normals: None,
        reverse_orientation: true,
        transform_swaps_handedness: false,
        emission_max: 4.0,
        scale: 0.5,
        two_sided: true,
    }])
    .unwrap();
    let triangle = bounds[0];
    assert_eq!(triangle.direction, [0.0, 0.0, -1.0]);
    assert_eq!(triangle.phi, 2.0 * std::f32::consts::PI);
    assert!(triangle.two_sided);
}

#[test]
fn input_normals_face_forward_without_orientation_flip() {
    let bounds = build_light_bounds(&[LightBoundInput::AreaTriangle {
        handle: 0,
        world_positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        input_normals: Some([[0.0, 0.0, -1.0]; 3]),
        reverse_orientation: false,
        transform_swaps_handedness: false,
        emission_max: 1.0,
        scale: 1.0,
        two_sided: false,
    }])
    .unwrap();
    assert_eq!(bounds[0].direction, [0.0, 0.0, -1.0]);
}

#[test]
fn invalid_input_normals_are_rejected() {
    let result = build_light_bounds(&[LightBoundInput::AreaTriangle {
        handle: 0,
        world_positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        input_normals: Some([[0.0, 0.0, 0.0]; 3]),
        reverse_orientation: false,
        transform_swaps_handedness: false,
        emission_max: 1.0,
        scale: 1.0,
        two_sided: false,
    }]);
    assert!(result.is_err());
}

#[test]
fn duplicate_handles_are_rejected() {
    let input = LightBoundInput::Point {
        handle: 1,
        world_position: [0.0; 3],
        intensity_max: 1.0,
        scale: 1.0,
    };
    assert!(build_light_bounds(&[input, input]).is_err());
}

#[test]
fn non_contiguous_handles_are_rejected() {
    assert!(build_light_bounds(&[LightBoundInput::Point {
        handle: 1,
        world_position: [0.0; 3],
        intensity_max: 1.0,
        scale: 1.0,
    }])
    .is_err());
}

#[test]
fn bounds_union_and_importance_are_validated() {
    let a = LightBounds {
        bounds: Bounds3::new([-1.0; 3], [1.0; 3]).unwrap(),
        direction: [0.0, 0.0, 1.0],
        phi: 1.0,
        cos_theta_o: 1.0,
        cos_theta_e: 0.0,
        two_sided: false,
    };
    let b = LightBounds {
        bounds: Bounds3::new([2.0; 3], [3.0; 3]).unwrap(),
        direction: [0.0, 0.0, 1.0],
        phi: 2.0,
        cos_theta_o: 1.0,
        cos_theta_e: 0.0,
        two_sided: false,
    };
    let union = a.union(b).unwrap();
    assert_eq!(union.bounds.min, [-1.0; 3]);
    assert_eq!(union.bounds.max, [3.0; 3]);
    assert!(union.importance([0.0, 0.0, 5.0], [0.0, 0.0, 1.0]).unwrap() >= 0.0);
}
