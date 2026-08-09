use std::collections::HashMap;
use std::sync::Arc;

use pbrt_r4::interaction::SurfaceInteraction;
use pbrt_r4::paramdict::{ParameterDictionary, TextureParameterDictionary};
use pbrt_r4::shapes::{Normal3f, Point2f, Point3f, Vector3f};
use pbrt_r4::textures::mapping2d::{PlanarMapping, TextureMapping2D};
use pbrt_r4::textures::TextureEvalContext;
use pbrt_r4::textures::{FloatTexture, SpectrumTexture};
use pbrt_r4::util::transform::Transform;

fn test_texture_eval_context() -> TextureEvalContext {
    let mut si = SurfaceInteraction::new(
        &Point3f::new(1.0, 1.0, 0.0),
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
    TextureEvalContext::from(&si)
}

#[test]
fn planar_mapping_applies_texture_transform_to_differentials() {
    let mapping = PlanarMapping::new(
        &Transform::scale(2.0, 3.0, 4.0),
        &Vector3f::new(1.0, 0.0, 0.0),
        &Vector3f::new(0.0, 1.0, 0.0),
        0.0,
        0.0,
    );
    let (st, dstdx, dstdy) = mapping.map(&test_texture_eval_context());

    assert_eq!(st, Point2f::new(2.0, 3.0));
    assert_eq!(dstdx, pbrt_r4::shapes::Vector2f::new(2.0, 0.0));
    assert_eq!(dstdy, pbrt_r4::shapes::Vector2f::new(0.0, 3.0));
}

#[test]
fn unknown_mapping_falls_back_to_uv_like_v4() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("mapping", "not-a-real-mapping");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let mapping = TextureMapping2D::create(&Transform::identity(), tp.parameter_dictionary())
        .expect("unknown mapping should fall back to UV");

    match mapping {
        TextureMapping2D::UV(_) => {}
        _ => panic!("expected UV fallback"),
    }
}
