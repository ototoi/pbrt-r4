use pbrt_r4::util::geometry::Vector3;
use pbrt_r4::util::transform::Matrix4x4;

type Vector3f = Vector3<f32>;

#[test]
fn test_001() {
    let m1 = Matrix4x4::scale(4.0, 4.0, 4.0);
    let m2 = m1.inverse().unwrap();
    let m3 = Matrix4x4::scale(0.25, 0.25, 0.25);
    assert_eq!(m2, m3);
}

#[test]
fn test_002() {
    let m1 = Matrix4x4::translate(4.0, 4.0, 4.0);
    let m2 = m1.inverse().unwrap();
    let m3 = Matrix4x4::translate(-4.0, -4.0, -4.0);
    assert_eq!(m2, m3);
}

#[test]
fn test_003() {
    let m1 = Matrix4x4::translate(4.0, 4.0, 4.0);
    let m2 = Matrix4x4::inverse(&m1).unwrap();
    let m3 = Matrix4x4::translate(-4.0, -4.0, -4.0);
    assert_eq!(m2, m3);
}

#[test]
fn test_004() {
    let m1 = Matrix4x4::rotate_x(90.0);
    let m2 = Matrix4x4::inverse(&m1).unwrap();
    let m3 = Matrix4x4::rotate_x(-90.0);
    assert_eq!(m2, m3);
}

#[test]
fn test_005() {
    let m1 = Matrix4x4::rotate_y(90.0);
    let m2 = Matrix4x4::inverse(&m1).unwrap();
    let m3 = Matrix4x4::rotate_y(-90.0);
    assert_eq!(m2, m3);
}

#[test]
fn test_006() {
    let m1 = Matrix4x4::rotate_z(90.0);
    let m2 = Matrix4x4::inverse(&m1).unwrap();
    let m3 = Matrix4x4::rotate_z(-90.0);
    assert_eq!(m2, m3);
}

#[test]
fn test_007() {
    let m1 = Matrix4x4::rotate_z(90.0);
    let m2 = Matrix4x4::transpose(&m1);
    let m3 = Matrix4x4::rotate_z(-90.0);
    assert_eq!(m2, m3);
}

#[test]
fn test_009() {
    let v1 = Vector3f::new(0.0, 0.0, 0.0);
    let m1 = Matrix4x4::look_at(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let m2 = m1.inverse().unwrap();
    let v2 = m2.transform_point(&v1);

    assert!(Vector3f::distance_squared(&v2, &Vector3f::new(0.0, 1.0, 1.0)) < 0.01);
}
