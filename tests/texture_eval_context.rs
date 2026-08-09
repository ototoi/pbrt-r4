use pbrt_r4::interaction::SurfaceInteraction;
use pbrt_r4::shapes::{Normal3f, Point2f, Point3f, Vector3f};
use pbrt_r4::textures::TextureEvalContext;

#[test]
fn texture_eval_context_copies_surface_interaction_fields() {
    let mut si = SurfaceInteraction::default();
    si.p = Point3f::new(1.0, 2.0, 3.0);
    si.dpdx = Vector3f::new(0.1, 0.2, 0.3);
    si.dpdy = Vector3f::new(0.4, 0.5, 0.6);
    si.n = Normal3f::new(0.0, 1.0, 0.0);
    si.uv = Point2f::new(0.25, 0.75);
    si.dudx = 1.0;
    si.dudy = 2.0;
    si.dvdx = 3.0;
    si.dvdy = 4.0;
    si.face_index = 7;

    let ctx = TextureEvalContext::from(&si);

    assert_eq!(ctx.p, si.p);
    assert_eq!(ctx.dpdx, si.dpdx);
    assert_eq!(ctx.dpdy, si.dpdy);
    assert_eq!(ctx.n, si.n);
    assert_eq!(ctx.uv, si.uv);
    assert_eq!(ctx.dudx, si.dudx);
    assert_eq!(ctx.dudy, si.dudy);
    assert_eq!(ctx.dvdx, si.dvdx);
    assert_eq!(ctx.dvdy, si.dvdy);
    assert_eq!(ctx.face_index, si.face_index);
}
