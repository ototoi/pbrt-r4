use std::sync::Arc;

use pbrt_r4::shapes::{Cone, Curve, CurveCommon, CurveType, Hyperboloid, Paraboloid, Sphere};
use pbrt_r4::util::base::{Normal3f, Point2f, Point3f, Vector3f};
use pbrt_r4::util::transform::Transform;

#[test]
fn cone_sampling_returns_valid_surface_sample() {
    let t = Transform::identity();
    let cone = Cone::new(&t, &t, false, 2.0, 1.0, 360.0);
    let (it, pdf) = cone.sample(&Point2f::new(0.3, 0.4)).unwrap();
    assert!(pdf.is_finite() && pdf > 0.0);
    assert!(it.get_n().length() > 0.0);
    assert!((0.0..=2.0).contains(&it.get_p().z));
}

#[test]
fn paraboloid_sampling_returns_valid_surface_sample() {
    let t = Transform::identity();
    let shape = Paraboloid::new(&t, &t, false, 1.0, 0.1, 1.0, 360.0);
    let (it, pdf) = shape.sample(&Point2f::new(0.25, 0.5)).unwrap();
    assert!(pdf.is_finite() && pdf > 0.0);
    assert!(it.get_n().length() > 0.0);
    assert!((0.1..=1.0).contains(&it.get_p().z));
}

#[test]
fn hyperboloid_sampling_returns_valid_pdf() {
    let t = Transform::identity();
    let shape = Hyperboloid::new(
        &t,
        &t,
        false,
        &Point3f::new(0.5, 0.0, -1.0),
        &Point3f::new(1.0, 0.0, 1.0),
        360.0,
    );
    let (it, pdf) = shape.sample(&Point2f::new(0.3, 0.7)).unwrap();
    assert!(pdf > 0.0 && pdf.is_finite());
    assert!(it.get_n().length() > 0.0);
}

#[test]
fn sphere_creation_and_intersection_match_expected_bounds() {
    let t = Transform::identity();
    let mut params = pbrt_r4::paramdict::ParameterDictionary::new();
    params.add_float("radius", 1.0);
    let sphere = Sphere::create(&t, &t, false, &params).unwrap();
    assert_eq!(sphere.radius, 1.0);
    let d = Vector3f::new(0.0, 0.0, 1.0);
    assert!(sphere.intersect_p(
        &pbrt_r4::util::geometry::Ray::new(&Point3f::new(0.0, 0.0, -5.0), &d, 1000.0, 0.0),
        1000.0
    ));
    assert!(!sphere.intersect_p(
        &pbrt_r4::util::geometry::Ray::new(&Point3f::new(0.0, 1.2, -5.0), &d, 1000.0, 0.0),
        1000.0
    ));
}

#[test]
fn flat_curve_sampling_returns_valid_interaction() {
    let t = Transform::identity();
    let cp = [
        Point3f::new(0.0, 0.0, 0.0),
        Point3f::new(0.3, 0.0, 0.3),
        Point3f::new(0.7, 0.0, 0.7),
        Point3f::new(1.0, 0.0, 1.0),
    ];
    let normals: Option<Vec<Normal3f>> = None;
    let common = Arc::new(CurveCommon::new(
        &t,
        &t,
        false,
        &cp,
        0.1,
        0.1,
        CurveType::Flat,
        &normals,
    ));
    let curve = Curve::new(&common, 0.0, 1.0);
    let (it, pdf) = curve.sample(&Point2f::new(0.3, 0.7)).unwrap();
    assert!(pdf > 0.0 && pdf.is_finite());
    assert!(it.get_n().length() > 0.0);
}
