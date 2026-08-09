use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

use std::sync::Arc;

pub struct MixTexture<T> {
    tex1: Arc<FloatTexture>,
    tex2: Arc<FloatTexture>,
    amount: Arc<FloatTexture>,
    _phantom: std::marker::PhantomData<T>,
}

impl MixTexture<Float> {
    pub fn new(
        tex1: &Arc<FloatTexture>,
        tex2: &Arc<FloatTexture>,
        amount: &Arc<FloatTexture>,
    ) -> Self {
        return MixTexture::<Float> {
            tex1: Arc::clone(tex1),
            tex2: Arc::clone(tex2),
            amount: Arc::clone(amount),
            _phantom: std::marker::PhantomData,
        };
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let t1 = self.tex1.evaluate(ctx);
        let t2 = self.tex2.evaluate(ctx);
        let amt = self.amount.evaluate(ctx);
        return t1 * (1.0 - amt) + t2 * amt;
    }

    pub fn create(
        _render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<FloatTexture, PbrtError> {
        let tex1 = parameters.get_float_texture("tex1", 0.0)?;
        let tex2 = parameters.get_float_texture("tex2", 1.0)?;
        let amount = parameters.get_float_texture("amount", 0.5)?;
        Ok(FloatTexture::Mix(MixTexture::<Float>::new(
            &tex1, &tex2, &amount,
        )))
    }
}

pub struct MixSpectrumTexture {
    tex1: Arc<SpectrumTexture>,
    tex2: Arc<SpectrumTexture>,
    amount: Arc<FloatTexture>,
}

pub struct DirectionMixFloatTexture {
    tex1: Arc<FloatTexture>,
    tex2: Arc<FloatTexture>,
    dir: Vector3f,
}

impl DirectionMixFloatTexture {
    pub fn new(tex1: &Arc<FloatTexture>, tex2: &Arc<FloatTexture>, dir: &Vector3f) -> Self {
        Self {
            tex1: Arc::clone(tex1),
            tex2: Arc::clone(tex2),
            dir: dir.normalize(),
        }
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let amount = Vector3f::dot(&ctx.n, &self.dir).abs();
        let t1 = self.tex1.evaluate(ctx);
        let t2 = self.tex2.evaluate(ctx);
        amount * t1 + (1.0 - amount) * t2
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<Self, PbrtError> {
        let tex1 = parameters.get_float_texture("tex1", 0.0)?;
        let tex2 = parameters.get_float_texture("tex2", 1.0)?;
        let dir = render_from_texture
            .transform_vector(&parameters.get_one_vector3f("dir", &Vector3f::new(0.0, 1.0, 0.0)));
        Ok(Self::new(&tex1, &tex2, &dir))
    }
}

pub struct DirectionMixSpectrumTexture {
    tex1: Arc<SpectrumTexture>,
    tex2: Arc<SpectrumTexture>,
    dir: Vector3f,
}

impl DirectionMixSpectrumTexture {
    pub fn new(tex1: &Arc<SpectrumTexture>, tex2: &Arc<SpectrumTexture>, dir: &Vector3f) -> Self {
        DirectionMixSpectrumTexture {
            tex1: Arc::clone(tex1),
            tex2: Arc::clone(tex2),
            dir: dir.normalize(),
        }
    }

    pub fn evaluate(
        &self,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        let amount = Vector3f::dot(&ctx.n, &self.dir).abs();
        let t1 = self.tex1.evaluate(ctx, lambda);
        let t2 = self.tex2.evaluate(ctx, lambda);
        amount * t1 + (1.0 - amount) * t2
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        let tex1 =
            parameters.get_spectrum_texture_typed("tex1", &Spectrum::zero(), spectrum_type)?;
        let tex2 =
            parameters.get_spectrum_texture_typed("tex2", &Spectrum::one(), spectrum_type)?;
        let dir = render_from_texture
            .transform_vector(&parameters.get_one_vector3f("dir", &Vector3f::new(0.0, 1.0, 0.0)));
        Ok(DirectionMixSpectrumTexture::new(&tex1, &tex2, &dir))
    }
}

impl MixSpectrumTexture {
    pub fn new(
        tex1: &Arc<SpectrumTexture>,
        tex2: &Arc<SpectrumTexture>,
        amount: &Arc<FloatTexture>,
    ) -> Self {
        return MixSpectrumTexture {
            tex1: Arc::clone(tex1),
            tex2: Arc::clone(tex2),
            amount: Arc::clone(amount),
        };
    }

    pub fn evaluate(
        &self,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        let t1 = self.tex1.evaluate(ctx, lambda);
        let t2 = self.tex2.evaluate(ctx, lambda);
        let amt = self.amount.evaluate(ctx);
        t1 * (1.0 - amt) + t2 * amt
    }

    pub fn create(
        _render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        let tex1 =
            parameters.get_spectrum_texture_typed("tex1", &Spectrum::one(), spectrum_type)?;
        let tex2 =
            parameters.get_spectrum_texture_typed("tex2", &Spectrum::zero(), spectrum_type)?;
        let amount = parameters.get_float_texture("amount", 0.5)?;
        Ok(MixSpectrumTexture::new(&tex1, &tex2, &amount))
    }
}
