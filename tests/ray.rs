use pbrt_r4::util::base::{Point3f, Vector3f};
use pbrt_r4::util::geometry::ray::Ray;

#[test]
fn position_matches_linear_ray_evaluation() {
    let ray = Ray::new(
        &Point3f::new(1.0, 2.0, 3.0),
        &Vector3f::new(1.0, 0.0, 0.0),
        1000.0,
        0.0,
    );
    assert_eq!(ray.position(4.0), Point3f::new(5.0, 2.0, 3.0));
}
