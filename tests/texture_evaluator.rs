use pbrt_r4::textures::texture_evaluator::{
    BasicTextureEvaluator, TextureEvaluator, UniversalTextureEvaluator,
};
use pbrt_r4::textures::TextureEvalContext;
use pbrt_r4::textures::{ConstantTexture, FloatTexture, RGBConstantSpectrumTexture, ScaleTexture};
use pbrt_r4::util::spectrum::{SampledWavelengths, Spectrum, SpectrumType};
use std::sync::Arc;

#[test]
fn universal_texture_evaluator_accepts_all_textures() {
    let c1 = Arc::new(FloatTexture::Constant(ConstantTexture::new(&2.0)));
    let c2 = Arc::new(FloatTexture::Constant(ConstantTexture::new(&3.0)));
    let scale = FloatTexture::Scale(ScaleTexture::new(&c1, &c2));
    let evaluator = UniversalTextureEvaluator;
    let ctx = TextureEvalContext::default();

    assert!(evaluator.can_evaluate(&[&scale], &[]));
    assert_eq!(evaluator.evaluate_float(&scale, &ctx), 6.0);
}

#[test]
fn basic_texture_evaluator_only_accepts_basic_textures() {
    let c1 = Arc::new(FloatTexture::Constant(ConstantTexture::new(&2.0)));
    let c2 = Arc::new(FloatTexture::Constant(ConstantTexture::new(&3.0)));
    let constant = FloatTexture::Constant(ConstantTexture::new(&4.0));
    let scale = FloatTexture::Scale(ScaleTexture::new(&c1, &c2));
    let spectrum = pbrt_r4::textures::SpectrumTexture::RGBConstant(
        RGBConstantSpectrumTexture::new(&[0.25, 0.5, 0.75], SpectrumType::Albedo),
    );
    let sampled =
        pbrt_r4::textures::SpectrumTexture::Constant(ConstantTexture::new(&Spectrum::from(0.5)));
    let lambda = SampledWavelengths::sample_visible(0.5);
    let evaluator = BasicTextureEvaluator;
    let ctx = TextureEvalContext::default();

    assert!(evaluator.can_evaluate(&[&constant], &[&spectrum, &sampled]));
    assert!(!evaluator.can_evaluate(&[&scale], &[]));
    assert_eq!(evaluator.evaluate_float(&constant, &ctx), 4.0);
    assert!(
        evaluator
            .evaluate_spectrum(&sampled, &ctx, &lambda)
            .average()
            > 0.0
    );
}
