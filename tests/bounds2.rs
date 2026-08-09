use pbrt_r4::util::geometry::{Bounds2f, Vector2};

type Vector2f = Vector2<f32>;

#[test]
fn test_001() {
    let v1 = Vector2f::new(1.0, 2.0);
    let v2 = Vector2f::new(4.0, 5.0);
    let b1 = Bounds2f::new(&v1, &v2);
    assert_eq!(b1.min, v1);
    assert_eq!(b1.max, v2);
}
