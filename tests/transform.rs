use pbrt_r4::util::base::Vector3f;
use pbrt_r4::util::transform::Transform;

#[test]
fn transform_inverse_matches_basic_ops() {
    assert_eq!(
        Transform::scale(4.0, 4.0, 4.0).inverse(),
        Transform::scale(0.25, 0.25, 0.25)
    );
    assert_eq!(
        Transform::translate(4.0, 4.0, 4.0).inverse(),
        Transform::translate(-4.0, -4.0, -4.0)
    );
    assert_eq!(
        Transform::rotate_x(90.0).inverse(),
        Transform::rotate_x(-90.0)
    );
    assert_eq!(
        Transform::rotate_y(90.0).inverse(),
        Transform::rotate_y(-90.0)
    );
    assert_eq!(
        Transform::rotate_z(90.0).inverse(),
        Transform::rotate_z(-90.0)
    );
}

#[test]
fn rotate_from_to_and_identity_behave_like_v4() {
    let from = Vector3f::new(1.0, 2.0, 3.0).normalize();
    let to = Vector3f::new(0.0, 0.0, 1.0);
    let r = Transform::rotate_from_to(from, to);
    assert!(Vector3f::distance_squared(&r.transform_vector(&from), &to) < 1e-6);
    assert!(Transform::identity().is_identity());
    assert!(!Transform::translate(1.0, 0.0, 0.0).is_identity());
}

#[test]
#[should_panic(
    expected = "LookAt: \"up\" vector and viewing direction passed to LookAt are pointing in the same direction."
)]
fn look_at_rejects_colinear_up_and_view_direction() {
    let _ = Transform::look_at(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0);
}
