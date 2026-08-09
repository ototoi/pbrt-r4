use pbrt_r4::util::geometry::Vector3;

type Vector3f = Vector3<f32>;
type Vector3d = Vector3<f64>;
type Vector3i = Vector3<i32>;

#[test]
fn vector3_add_sub_mul_and_scalars_work() {
    let v1 = Vector3f::new(1.0, 2.0, 3.0);
    let v2 = Vector3f::new(4.0, 5.0, 6.0);
    assert_eq!(v1 + v2, Vector3f::new(5.0, 7.0, 9.0));
    assert_eq!(v2 - v1, Vector3f::new(3.0, 3.0, 3.0));
    assert_eq!(v2 * v1, Vector3f::new(4.0, 10.0, 18.0));
    assert_eq!(2.0 * v2, Vector3f::new(8.0, 10.0, 12.0));
}

#[test]
fn vector3_abs_length_and_normalize_work() {
    assert_eq!(
        Vector3f::new(-1.0, 2.0, -3.0).abs(),
        Vector3f::new(1.0, 2.0, 3.0)
    );
    assert_eq!(
        Vector3d::new(-1.0, 2.0, -3.0).abs(),
        Vector3d::new(1.0, 2.0, 3.0)
    );
    assert_eq!(Vector3i::new(-1, 2, -3).abs(), Vector3i::new(1, 2, 3));
    let v = Vector3f::new(4.0, 0.0, 0.0);
    assert_eq!(v.length(), 4.0);
    assert_eq!(v.length_squared(), 16.0);
    assert_eq!(v.normalize(), Vector3f::new(1.0, 0.0, 0.0));
}

#[test]
fn vector3_from_conversions_work() {
    let v1 = Vector3f::new(1.0, 2.0, 3.0);
    assert_eq!(v1, Vector3f::from((1.0, 2.0, 3.0)));
    assert_eq!(v1, Vector3f::from([1.0, 2.0, 3.0]));
}
