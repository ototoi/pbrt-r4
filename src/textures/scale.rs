use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::sync::Arc;

pub struct ScaleTexture<T1, T2> {
    tex1: Arc<FloatTexture>,
    tex2: Arc<FloatTexture>,
    _phantom: std::marker::PhantomData<(T1, T2)>,
}

impl ScaleTexture<Float, Float> {
    pub fn new(tex1: &Arc<FloatTexture>, tex2: &Arc<FloatTexture>) -> Self {
        return ScaleTexture::<Float, Float> {
            tex1: Arc::clone(tex1),
            tex2: Arc::clone(tex2),
            _phantom: std::marker::PhantomData,
        };
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let value1 = self.tex1.evaluate(ctx);
        let value2 = self.tex2.evaluate(ctx);
        return value1 * value2;
    }

    pub fn create(
        _render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<FloatTexture, PbrtError> {
        let tex = parameters
            .get_float_texture_or_null("tex")?
            .or(parameters.get_float_texture_or_null("tex1")?)
            .ok_or_else(|| {
                PbrtError::error(
                    "Scale texture requires a float texture named \"tex\" or \"tex1\".",
                )
            })?;
        let scale = parameters
            .get_float_texture_or_null("scale")?
            .or(parameters.get_float_texture_or_null("tex2")?)
            .unwrap_or_else(|| Arc::new(FloatTexture::Constant(ConstantTexture::new(&1.0))));
        Ok(FloatTexture::Scale(ScaleTexture::<Float, Float>::new(
            &tex, &scale,
        )))
    }
}

enum ScaleSpectrumTextureMode {
    SpectrumTimesSpectrum {
        tex1: Arc<SpectrumTexture>,
        tex2: Arc<SpectrumTexture>,
    },
    SpectrumTimesFloat {
        tex: Arc<SpectrumTexture>,
        scale: Arc<FloatTexture>,
    },
}

pub struct ScaleSpectrumTexture {
    mode: ScaleSpectrumTextureMode,
}

impl ScaleSpectrumTexture {
    pub fn new(tex1: &Arc<SpectrumTexture>, tex2: &Arc<SpectrumTexture>) -> Self {
        return ScaleSpectrumTexture {
            mode: ScaleSpectrumTextureMode::SpectrumTimesSpectrum {
                tex1: Arc::clone(tex1),
                tex2: Arc::clone(tex2),
            },
        };
    }

    pub fn new_v4(tex: &Arc<SpectrumTexture>, scale: &Arc<FloatTexture>) -> Self {
        return ScaleSpectrumTexture {
            mode: ScaleSpectrumTextureMode::SpectrumTimesFloat {
                tex: Arc::clone(tex),
                scale: Arc::clone(scale),
            },
        };
    }

    /// v4 verbatim hot-path entry — keeps the arithmetic in
    /// `SampledSpectrum` so the per-shade `Spectrum::DenselySampled`
    /// alloc the `Spectrum::Mul` operator forces never happens.
    pub fn evaluate(
        &self,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        match &self.mode {
            ScaleSpectrumTextureMode::SpectrumTimesSpectrum { tex1, tex2 } => {
                tex1.evaluate(ctx, lambda) * tex2.evaluate(ctx, lambda)
            }
            ScaleSpectrumTextureMode::SpectrumTimesFloat { tex, scale } => {
                tex.evaluate(ctx, lambda) * scale.evaluate(ctx)
            }
        }
    }

    pub fn create(
        _render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        let tex = parameters
            .get_spectrum_texture_or_null_typed("tex", spectrum_type)?
            .or(parameters.get_spectrum_texture_or_null_typed("tex1", spectrum_type)?)
            .ok_or_else(|| {
                PbrtError::error(
                    "Scale texture requires a spectrum texture named \"tex\" or \"tex1\".",
                )
            })?;
        let scale = parameters
            .get_float_texture_or_null("scale")?
            .or(parameters.get_float_texture_or_null("tex2")?)
            .unwrap_or_else(|| Arc::new(FloatTexture::Constant(ConstantTexture::new(&1.0))));
        Ok(ScaleSpectrumTexture::new_v4(&tex, &scale))
    }
}
