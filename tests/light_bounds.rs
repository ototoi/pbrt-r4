use pbrt_r4::base::light::{union_light_bounds, LightBounds};
use pbrt_r4::prelude::*;

fn forward_light_bounds() -> LightBounds {
    LightBounds::new(
        Bounds3f::new(
            &Point3f::new(-0.5, -0.5, -0.05),
            &Point3f::new(0.5, 0.5, 0.05),
        ),
        Vector3f::new(0.0, 0.0, 1.0),
        10.0,
        1.0,
        0.0,
        false,
    )
}

#[test]
fn light_bounds_importance_accounts_for_direction_and_distance() {
    let bounds = forward_light_bounds();
    let near = bounds.importance(Point3f::new(0.0, 0.0, 2.0), Normal3f::zero());
    let far = bounds.importance(Point3f::new(0.0, 0.0, 4.0), Normal3f::zero());
    let behind = bounds.importance(Point3f::new(0.0, 0.0, -2.0), Normal3f::zero());

    assert!(near > far);
    assert_eq!(behind, 0.0);
}

#[test]
fn union_light_bounds_unites_direction_cones() {
    let left = LightBounds::new(
        Bounds3f::new(&Point3f::zero(), &Point3f::zero()),
        Vector3f::new(-1.0, 0.0, 1.0),
        1.0,
        1.0,
        0.0,
        false,
    );
    let right = LightBounds::new(
        Bounds3f::new(&Point3f::zero(), &Point3f::zero()),
        Vector3f::new(1.0, 0.0, 1.0),
        1.0,
        1.0,
        0.0,
        false,
    );

    let united = union_light_bounds(&left, &right);
    assert!(united.w.x.abs() < 1e-5);
    assert!(united.w.z > 0.99);
    assert!(united.cos_theta_o < 1.0);
}
