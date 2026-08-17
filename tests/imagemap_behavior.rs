use pbrt_r4::textures::{
    FloatImageTexture, ImageFilter, ImageWrap, MIPMap, SpectrumImageTexture, TextureEvalContext,
    TextureMapping2D, UVMapping,
};
use pbrt_r4::util::base::{Float, Point2f, Point2i};
use pbrt_r4::util::spectrum::{RGBSpectrum, SampledWavelengths, SpectrumType};

#[test]
fn float_imagemap_applies_scale_after_filtering() {
    let mipmap = MIPMap::<Float>::new(
        &Point2i::new(3, 3),
        &vec![0.5; 9],
        ImageFilter::Bilinear,
        8.0,
        ImageWrap::Repeat,
        ImageWrap::Repeat,
    );
    let texture = FloatImageTexture::new(
        TextureMapping2D::UV(UVMapping::new(1.0, 1.0, 0.0, 0.0)),
        mipmap,
        -2.0,
        false,
    );
    let mut ctx = TextureEvalContext::default();
    ctx.uv = Point2f::new(0.5, 0.5);
    assert!((texture.evaluate(&ctx) + 1.0).abs() < 1e-4);
}

#[test]
fn float_imagemap_flips_t_at_evaluation_like_v4() {
    let mipmap = MIPMap::<Float>::new(
        &Point2i::new(1, 2),
        &vec![0.0, 1.0],
        ImageFilter::Point,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
    );
    let texture = FloatImageTexture::new(
        TextureMapping2D::UV(UVMapping::new(1.0, 1.0, 0.0, 0.0)),
        mipmap,
        1.0,
        false,
    );
    let mut ctx = TextureEvalContext::default();
    ctx.uv = Point2f::new(0.5, 0.25);
    let lower = texture.evaluate(&ctx);
    ctx.uv[1] = 0.75;
    let upper = texture.evaluate(&ctx);

    assert!(lower > upper);
}

#[test]
fn spectrum_imagemap_flips_t_at_evaluation_like_v4() {
    let mipmap = MIPMap::<RGBSpectrum>::new(
        &Point2i::new(1, 2),
        &vec![
            RGBSpectrum::new(0.0, 0.0, 0.0),
            RGBSpectrum::new(1.0, 1.0, 1.0),
        ],
        ImageFilter::Point,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
    );
    let texture = SpectrumImageTexture::new_shared(
        TextureMapping2D::UV(UVMapping::new(1.0, 1.0, 0.0, 0.0)),
        std::sync::Arc::new(mipmap),
        SpectrumType::Albedo,
        1.0,
        false,
    );
    let lambda = SampledWavelengths::sample_visible(0.5);
    let mut ctx = TextureEvalContext::default();
    ctx.uv = Point2f::new(0.5, 0.25);
    let lower = texture.evaluate(&ctx, &lambda).max_component_value();
    ctx.uv[1] = 0.75;
    let upper = texture.evaluate(&ctx, &lambda).max_component_value();

    assert!(lower > upper);
}
