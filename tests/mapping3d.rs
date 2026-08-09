use pbrt_r4::interaction::SurfaceInteraction;
use pbrt_r4::shapes::{Normal3f, Point2f, Point3f, Vector3f};
use pbrt_r4::textures::mapping3d::PointTransformMapping;
use pbrt_r4::textures::TextureEvalContext;
use pbrt_r4::util::transform::Transform;

#[test]
fn identity_mapping_uses_surface_dpdy_for_dpdy() {
    let mapping = PointTransformMapping::new(&Transform::scale(2.0, 3.0, 4.0));
    let mut si = SurfaceInteraction::new(
        &Point3f::new(0.0, 0.0, 0.0),
        &Point3f::zero(),
        &Point2f::new(0.0, 0.0),
        &Vector3f::new(0.0, 0.0, 1.0),
        &Normal3f::new(0.0, 0.0, 1.0),
        &Vector3f::new(1.0, 0.0, 0.0),
        &Vector3f::new(0.0, 1.0, 0.0),
        &Vector3f::zero(),
        &Vector3f::zero(),
        0.0,
        0,
    );
    si.dpdx = Vector3f::new(1.0, 0.0, 0.0);
    si.dpdy = Vector3f::new(0.0, 1.0, 0.0);
    let ctx = TextureEvalContext::from(&si);

    let (_, dpdx, dpdy) = mapping.map(&ctx);

    assert_eq!(dpdx, Vector3f::new(2.0, 0.0, 0.0));
    assert_eq!(dpdy, Vector3f::new(0.0, 3.0, 0.0));
}
