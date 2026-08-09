use pbrt_r4::util::transform::{Transform, TransformSet};

#[test]
fn test_001() {
    let mut ts = TransformSet::new();
    assert_eq!(ts[0], Transform::identity());
    ts[1] = Transform::translate(-10.0, 0.0, 2.0);
    assert_ne!(ts[1], Transform::identity());
    assert_eq!(ts[1], Transform::translate(-10.0, 0.0, 2.0));
}
