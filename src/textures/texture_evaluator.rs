use crate::base::texture::{FloatTexture, SpectrumTexture};
use crate::textures::texture_eval_context::TextureEvalContext;
use crate::util::base::Float;
use crate::util::spectrum::{SampledSpectrum, SampledWavelengths};

pub trait TextureEvaluator {
    fn can_evaluate(
        &self,
        float_textures: &[&FloatTexture],
        spectrum_textures: &[&SpectrumTexture],
    ) -> bool;

    fn evaluate_float(&self, texture: &FloatTexture, ctx: &TextureEvalContext) -> Float;

    fn evaluate_spectrum(
        &self,
        texture: &SpectrumTexture,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum;
}

#[derive(Copy, Clone, Debug, Default)]
pub struct UniversalTextureEvaluator;

impl TextureEvaluator for UniversalTextureEvaluator {
    fn can_evaluate(
        &self,
        _float_textures: &[&FloatTexture],
        _spectrum_textures: &[&SpectrumTexture],
    ) -> bool {
        true
    }

    fn evaluate_float(&self, texture: &FloatTexture, ctx: &TextureEvalContext) -> Float {
        texture.evaluate(ctx)
    }

    fn evaluate_spectrum(
        &self,
        texture: &SpectrumTexture,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        texture.evaluate(ctx, lambda)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BasicTextureEvaluator;

impl BasicTextureEvaluator {
    fn can_evaluate_float(texture: &FloatTexture) -> bool {
        matches!(
            texture,
            FloatTexture::Constant(_) | FloatTexture::ImageMap(_)
        )
    }

    fn can_evaluate_spectrum(texture: &SpectrumTexture) -> bool {
        matches!(
            texture,
            SpectrumTexture::Constant(_)
                | SpectrumTexture::RGBConstant(_)
                | SpectrumTexture::SampledCurve(_)
                | SpectrumTexture::BlackbodyConstant(_)
                | SpectrumTexture::ImageMap(_)
        )
    }
}

impl TextureEvaluator for BasicTextureEvaluator {
    fn can_evaluate(
        &self,
        float_textures: &[&FloatTexture],
        spectrum_textures: &[&SpectrumTexture],
    ) -> bool {
        float_textures
            .iter()
            .all(|tex| Self::can_evaluate_float(tex))
            && spectrum_textures
                .iter()
                .all(|tex| Self::can_evaluate_spectrum(tex))
    }

    fn evaluate_float(&self, texture: &FloatTexture, ctx: &TextureEvalContext) -> Float {
        match texture {
            FloatTexture::Constant(_) | FloatTexture::ImageMap(_) => texture.evaluate(ctx),
            _ => unreachable!(
                "BasicTextureEvaluator::evaluate_float called with unsupported texture"
            ),
        }
    }

    fn evaluate_spectrum(
        &self,
        texture: &SpectrumTexture,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        match texture {
            SpectrumTexture::Constant(_)
            | SpectrumTexture::RGBConstant(_)
            | SpectrumTexture::SampledCurve(_)
            | SpectrumTexture::BlackbodyConstant(_)
            | SpectrumTexture::ImageMap(_) => texture.evaluate(ctx, lambda),
            _ => unreachable!(
                "BasicTextureEvaluator::evaluate_spectrum called with unsupported texture"
            ),
        }
    }
}
