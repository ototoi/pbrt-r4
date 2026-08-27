use pbrt_r4::util::base::Point3f;
use pbrt_r4::util::geometry::{bound_subtended_directions, Bounds3f, DirectionCone};

#[test]
fn degenerate_bounds_at_reference_point_cover_the_sphere() {
    let p = Point3f::new(1.0, 2.0, 3.0);
    let bounds = Bounds3f::from((p.x, p.y, p.z));

    let cone = bound_subtended_directions(&bounds, &p);

    assert_eq!(cone, DirectionCone::entire_sphere());
    assert!(cone.w.x.is_finite() && cone.w.y.is_finite() && cone.w.z.is_finite());
}

#[test]
fn reference_point_inside_bounding_sphere_covers_the_sphere() {
    let bounds = Bounds3f::new(&Point3f::new(-1.0, 0.0, 0.0), &Point3f::new(1.0, 0.0, 0.0));

    let cone = bound_subtended_directions(&bounds, &Point3f::new(0.0, 0.0, 0.0));

    assert_eq!(cone, DirectionCone::entire_sphere());
}

#[test]
fn reference_point_on_bounding_sphere_returns_a_hemisphere_cone() {
    let bounds = Bounds3f::new(&Point3f::new(-1.0, 0.0, 0.0), &Point3f::new(1.0, 0.0, 0.0));

    let cone = bound_subtended_directions(&bounds, &Point3f::new(1.0, 0.0, 0.0));

    assert!((cone.cos_theta).abs() < 1e-6);
    assert!(cone.w.x.is_finite() && cone.w.y.is_finite() && cone.w.z.is_finite());
}
