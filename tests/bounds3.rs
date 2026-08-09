use pbrt_r4::util::base::Vector3f;
use pbrt_r4::util::geometry::Bounds3f;

#[test]
fn bounds3_new_orders_corners() {
    let v1 = Vector3f::new(1.0, 2.0, 3.0);
    let v2 = Vector3f::new(4.0, 5.0, 6.0);
    let b = Bounds3f::new(&v2, &v1);
    assert_eq!(b.min, v1);
    assert_eq!(b.max, v2);
}
