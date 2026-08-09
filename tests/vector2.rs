use pbrt_r4::util::geometry::Vector2;

type Vector2f = Vector2<f32>;

#[test]
fn test_001() {
    let v1 = Vector2f::new(1.0, 2.0);
    let v2 = Vector2f::new(3.0, 4.0);
    let v3 = v1 + v2;
    let v4 = Vector2f::new(4.0, 6.0);
    assert_eq!(v3, v4);
}

#[test]
fn test_002() {
    let v1 = Vector2f::new(1.0, 2.0);
    let v2 = Vector2f::new(3.0, 4.0);
    let v3 = v2 - v1;
    let v4 = Vector2f::new(2.0, 2.0);
    assert_eq!(v3, v4);
}

#[test]
fn test_003() {
    let v1 = Vector2f::new(1.0, 2.0);
    let v2 = Vector2f::new(3.0, 4.0);
    let v3 = v2 * v1;
    let v4 = Vector2f::new(3.0, 8.0);
    assert_eq!(v3, v4);
}
