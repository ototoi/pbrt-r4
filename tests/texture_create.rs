use pbrt_r4::base::typed_spectrum_texture_name;
use pbrt_r4::prelude::*;

use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn spectrum_texture_create_directionmix_succeeds() {
    let mut geom_params = ParameterDictionary::new();
    geom_params.add_vector3f("dir", &Vector3f::new(0.0, 1.0, 0.0));

    let _mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&geom_params, &f_tex, &s_tex);

    let tex = SpectrumTexture::create(
        "directionmix",
        &Transform::identity(),
        &tp,
        SpectrumType::Albedo,
    );
    assert!(tex.is_ok());

    let tex = tex.unwrap();
    let ctx = TextureEvalContext::default();
    let lambda = SampledWavelengths::sample_visible(0.5);
    let c = tex.evaluate(&ctx, &lambda);
    assert!(!c.is_black());
}

#[test]
fn float_texture_create_directionmix_succeeds() {
    let mut geom_params = ParameterDictionary::new();
    geom_params.add_vector3f("dir", &Vector3f::new(0.0, 1.0, 0.0));
    let _mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&geom_params, &f_tex, &s_tex);

    let tex = FloatTexture::create("directionmix", &Transform::identity(), &tp).unwrap();
    let mut ctx = TextureEvalContext::default();
    ctx.n = Vector3f::new(1.0, 0.0, 0.0);
    assert!((tex.evaluate(&ctx) - 1.0).abs() < 1e-6);
}

#[test]
fn float_texture_create_unknown_type_returns_error() {
    let _geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(FloatTexture::create("not-a-real-texture", &Transform::identity(), &tp).is_err());
}

#[test]
fn float_mix_texture_defaults_match_v4_tex1_tex2_parameters() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_float("amount", 0.0);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let tex = FloatTexture::create("mix", &Transform::identity(), &tp).unwrap();
    let value = tex.evaluate(&TextureEvalContext::default());
    assert!((value - 0.0).abs() < 1e-6);
}

#[test]
fn float_scale_texture_supports_v4_tex_times_scale_parameters() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture tex", "base");
    mat_params.add_float("scale", 2.5);

    let mut f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    f_tex.insert(
        "base".to_string(),
        Arc::new(FloatTexture::Constant(ConstantTexture::new(&0.4))),
    );
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let tex = FloatTexture::create("scale", &Transform::identity(), &tp).unwrap();
    let value = tex.evaluate(&TextureEvalContext::default());
    assert!((value - 1.0).abs() < 1e-6);
}

#[test]
fn float_scale_texture_defaults_to_one_like_v4() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture tex", "base");
    let mut f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    f_tex.insert(
        "base".to_string(),
        Arc::new(FloatTexture::Constant(ConstantTexture::new(&0.4))),
    );
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let tex = FloatTexture::create("scale", &Transform::identity(), &tp).unwrap();
    let value = tex.evaluate(&TextureEvalContext::default());
    assert!((value - 0.4).abs() < 1e-6);
}

#[test]
fn scale_texture_accepts_v4_legacy_tex1_tex2_aliases() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture tex1", "base");
    mat_params.add_float("scale", 2.5);

    let mut f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    f_tex.insert(
        "base".to_string(),
        Arc::new(FloatTexture::Constant(ConstantTexture::new(&0.4))),
    );
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let tex = FloatTexture::create("scale", &Transform::identity(), &tp).unwrap();
    let value = tex.evaluate(&TextureEvalContext::default());
    assert!((value - 1.0).abs() < 1e-6);
}

#[test]
fn spectrum_scale_texture_accepts_v4_legacy_tex1_tex2_aliases() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture tex1", "base");
    mat_params.add_float("scale", 2.5);

    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let mut s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    s_tex.insert(
        "base".to_string(),
        Arc::new(SpectrumTexture::Constant(ConstantTexture::new(
            &Spectrum::from(0.4),
        ))),
    );
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let tex = SpectrumTexture::create("scale", &Transform::identity(), &tp, SpectrumType::Albedo)
        .unwrap();
    let ctx = TextureEvalContext::default();
    let lambda = SampledWavelengths::sample_visible(0.5);
    let value = tex.evaluate(&ctx, &lambda);
    assert!((value.max_component_value() - 1.0).abs() < 1e-6);
}

#[test]
fn scale_texture_errors_when_named_tex_cannot_be_resolved() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture tex", "missing");
    mat_params.add_float("scale", 2.5);

    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(FloatTexture::create("scale", &Transform::identity(), &tp).is_err());
}

#[test]
fn mix_texture_defaults_match_v4_tex1_tex2_parameters() {
    let _geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let tex = FloatTexture::create("mix", &Transform::identity(), &tp).unwrap();
    let value = tex.evaluate(&TextureEvalContext::default());
    assert!((value - 0.5).abs() < 1e-6);
}

#[test]
fn spectrum_mix_texture_defaults_match_v4_tex1_tex2_parameters() {
    let _geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let tex =
        SpectrumTexture::create("mix", &Transform::identity(), &tp, SpectrumType::Albedo).unwrap();
    let ctx = TextureEvalContext::default();
    let lambda = SampledWavelengths::sample_visible(0.5);
    let value = tex.evaluate(&ctx, &lambda);
    assert!((value.max_component_value() - 0.5).abs() < 1e-6);
}

#[test]
fn checkerboard_defaults_match_v4_tex1_tex2_and_closedform_aamode() {
    let _geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let tex = FloatTexture::create("checkerboard", &Transform::identity(), &tp).unwrap();
    let ctx = TextureEvalContext::default();
    assert!((tex.evaluate(&ctx) - 1.0).abs() < 1e-6);
}

#[test]
fn checkerboard_unknown_aamode_returns_error() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("aamode", "not-a-real-mode");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(FloatTexture::create("checkerboard", &Transform::identity(), &tp).is_err());
}

#[test]
fn fbm_defaults_match_v4_octaves_and_roughness() {
    let _geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    explicit_params.add_int("octaves", 8);
    explicit_params.add_float("roughness", 0.5);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);

    let default = FloatTexture::create("fbm", &Transform::identity(), &default_tp)
        .expect("default fbm should create");
    let explicit = FloatTexture::create("fbm", &Transform::identity(), &explicit_tp)
        .expect("explicit fbm should create");

    let ctx = TextureEvalContext::default();
    assert!((default.evaluate(&ctx) - explicit.evaluate(&ctx)).abs() < 1e-6);
}

#[test]
fn wrinkled_defaults_match_v4_octaves_and_roughness() {
    let _geom_params = ParameterDictionary::new();
    let default_params = ParameterDictionary::new();
    let mut explicit_params = ParameterDictionary::new();
    explicit_params.add_int("octaves", 8);
    explicit_params.add_float("roughness", 0.5);
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let default_tp = TextureParameterDictionary::new(&default_params, &f_tex, &s_tex);
    let explicit_tp = TextureParameterDictionary::new(&explicit_params, &f_tex, &s_tex);

    let default = FloatTexture::create("wrinkled", &Transform::identity(), &default_tp)
        .expect("default wrinkled should create");
    let explicit = FloatTexture::create("wrinkled", &Transform::identity(), &explicit_tp)
        .expect("explicit wrinkled should create");

    let ctx = TextureEvalContext::default();
    assert!((default.evaluate(&ctx) - explicit.evaluate(&ctx)).abs() < 1e-6);
}

#[test]
fn imagemap_png_defaults_to_v4_filter_and_gamma() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("filename", "textures/example_bump.png");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let texinfo = create_texinfo(&tp)
        .expect("texinfo creation should succeed")
        .expect("filename should produce texinfo");
    assert_eq!(texinfo.filter, ImageFilter::Bilinear);
    assert_eq!(texinfo.swrap_mode, ImageWrap::Repeat);
    assert_eq!(texinfo.twrap_mode, ImageWrap::Repeat);
    assert!(texinfo.gamma);
}

#[test]
fn imagemap_missing_filename_returns_none() {
    let _geom_params = ParameterDictionary::new();
    let mat_params = ParameterDictionary::new();
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let texinfo = create_texinfo(&tp).expect("texinfo creation should succeed");
    assert!(texinfo.is_none());
}

#[test]
fn imagemap_invalid_wrap_and_filter_return_errors() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("filename", "textures/example_bump.png");
    mat_params.add_string("wrap", "not-a-wrap");
    mat_params.add_string("filter", "not-a-filter");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    assert!(create_texinfo(&tp).is_err());
}

#[test]
fn imagemap_supports_v4_filter_and_encoding_parameters() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("filename", "textures/example_bump.png");
    mat_params.add_string("filter", "ewa");
    mat_params.add_string("encoding", "linear");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let texinfo = create_texinfo(&tp)
        .expect("texinfo creation should succeed")
        .expect("filename should produce texinfo");
    assert_eq!(texinfo.filter, ImageFilter::EWA);
    assert!(!texinfo.gamma);
}

#[test]
fn imagemap_exr_defaults_to_linear_encoding_like_v4() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("filename", "textures/example_bump.exr");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let texinfo = create_texinfo(&tp)
        .expect("texinfo creation should succeed")
        .expect("filename should produce texinfo");
    assert!(!texinfo.gamma);
}

#[test]
fn texture_parameter_dictionary_named_rgb_textures_are_type_partitioned() {
    let _geom_params = ParameterDictionary::new();
    let mut mat_params = ParameterDictionary::new();
    mat_params.add_string("texture reflectance", "paint");
    mat_params.add_string("texture eta", "paint");

    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let mut s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    s_tex.insert(
        typed_spectrum_texture_name("paint", SpectrumType::Albedo),
        Arc::new(spectrum_texture_from_rgb_constant(
            &[0.8, 0.2, 0.1],
            SpectrumType::Albedo,
        )),
    );
    s_tex.insert(
        typed_spectrum_texture_name("paint", SpectrumType::Unbounded),
        Arc::new(spectrum_texture_from_rgb_constant(
            &[2.0, 2.0, 2.0],
            SpectrumType::Unbounded,
        )),
    );
    let tp = TextureParameterDictionary::new(&mat_params, &f_tex, &s_tex);

    let reflectance = tp
        .get_spectrum_texture_or_null_typed("reflectance", SpectrumType::Albedo)
        .expect("typed albedo texture should resolve");
    let reflectance = reflectance.expect("reflectance texture should be present");
    let eta = tp
        .get_spectrum_texture_or_null_typed("eta", SpectrumType::Unbounded)
        .expect("typed unbounded texture should resolve");
    let eta = eta.expect("eta texture should be present");

    let ctx = TextureEvalContext::default();
    let lambda = SampledWavelengths::sample_visible(0.5);
    assert!(reflectance.evaluate(&ctx, &lambda).max_component_value() <= 1.0 + 1e-6);
    assert!(eta.evaluate(&ctx, &lambda).max_component_value() > 1.0);
}
