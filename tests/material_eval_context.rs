use pbrt_r4::interaction::SurfaceInteraction;
use pbrt_r4::materials::MaterialEvalContext;
use pbrt_r4::prelude::*;

#[test]
fn material_eval_context_copies_surface_interaction_fields() {
    let mut si = SurfaceInteraction::default();
    si.p = Point3f::new(1.0, 2.0, 3.0);
    si.uv = Point2f::new(0.25, 0.75);
    si.wo = Vector3f::new(0.0, 0.0, 1.0);
    si.n = Normal3f::new(0.0, 1.0, 0.0);
    si.shading.n = Normal3f::new(1.0, 0.0, 0.0);
    si.shading.dpdu = Vector3f::new(0.5, 0.25, 0.0);
    si.dpdx = Vector3f::new(0.1, 0.2, 0.3);
    si.dpdy = Vector3f::new(0.4, 0.5, 0.6);
    si.dudx = 1.0;
    si.dudy = 2.0;
    si.dvdx = 3.0;
    si.dvdy = 4.0;
    si.face_index = 7;

    let ctx = MaterialEvalContext::from(&si);

    assert_eq!(ctx.texture_ctx.p, si.p);
    assert_eq!(ctx.texture_ctx.uv, si.uv);
    assert_eq!(ctx.texture_ctx.dpdx, si.dpdx);
    assert_eq!(ctx.texture_ctx.dpdy, si.dpdy);
    assert_eq!(ctx.texture_ctx.face_index, si.face_index);
    assert_eq!(ctx.wo, si.wo);
    assert_eq!(ctx.ns, si.shading.n);
    assert_eq!(ctx.dpdus, si.shading.dpdu);
}
