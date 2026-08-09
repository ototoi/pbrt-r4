use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

use std::sync::Arc;

pub struct DotsTexture<T> {
    mapping: TextureMapping2D,
    outside_dot: Arc<FloatTexture>,
    inside_dot: Arc<FloatTexture>,
    _phantom: std::marker::PhantomData<T>,
}

impl DotsTexture<Float> {
    pub fn new(
        mapping: TextureMapping2D,
        tex1: &Arc<FloatTexture>,
        tex2: &Arc<FloatTexture>,
    ) -> Self {
        return DotsTexture::<Float> {
            mapping,
            outside_dot: Arc::clone(tex1),
            inside_dot: Arc::clone(tex2),
            _phantom: std::marker::PhantomData,
        };
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let (st, _dstdx, _dstdy) = self.mapping.map(ctx);
        let s_cell = Float::floor(st[0] + 0.5);
        let t_cell = Float::floor(st[1] + 0.5);
        if noise(s_cell + 0.5, t_cell + 0.5, 0.0) > 0.0 {
            let radius = 0.35;
            let max_shift = 0.5 - radius;
            let s_center = s_cell + max_shift * noise(s_cell + 1.5, t_cell + 2.8, 0.0);
            let t_center = t_cell + max_shift * noise(s_cell + 4.5, t_cell + 9.8, 0.0);
            let dst = st - Point2f::new(s_center, t_center);
            if dst.length_squared() < radius * radius {
                return self.inside_dot.evaluate(ctx);
            }
        }
        return self.outside_dot.evaluate(ctx);
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<FloatTexture, PbrtError> {
        let map = TextureMapping2D::create(render_from_texture, parameters.parameter_dictionary())?;
        let tex1 = parameters.get_float_texture("tex1", 1.0)?;
        let tex2 = parameters.get_float_texture("tex2", 0.0)?;
        Ok(FloatTexture::Dots(DotsTexture::<Float>::new(
            map, &tex1, &tex2,
        )))
    }
}

pub struct DotsSpectrumTexture {
    mapping: TextureMapping2D,
    outside_dot: Arc<SpectrumTexture>,
    inside_dot: Arc<SpectrumTexture>,
}

impl DotsSpectrumTexture {
    pub fn new(
        mapping: TextureMapping2D,
        tex1: &Arc<SpectrumTexture>,
        tex2: &Arc<SpectrumTexture>,
    ) -> Self {
        return DotsSpectrumTexture {
            mapping,
            outside_dot: Arc::clone(tex1),
            inside_dot: Arc::clone(tex2),
        };
    }

    pub fn evaluate(
        &self,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        if self.inside(ctx) {
            self.inside_dot.evaluate(ctx, lambda)
        } else {
            self.outside_dot.evaluate(ctx, lambda)
        }
    }

    fn inside(&self, ctx: &TextureEvalContext) -> bool {
        let (st, _dstdx, _dstdy) = self.mapping.map(ctx);
        let s_cell = Float::floor(st[0] + 0.5);
        let t_cell = Float::floor(st[1] + 0.5);
        if noise(s_cell + 0.5, t_cell + 0.5, 0.0) > 0.0 {
            let radius = 0.35;
            let max_shift = 0.5 - radius;
            let s_center = s_cell + max_shift * noise(s_cell + 1.5, t_cell + 2.8, 0.0);
            let t_center = t_cell + max_shift * noise(s_cell + 4.5, t_cell + 9.8, 0.0);
            let dst = st - Point2f::new(s_center, t_center);
            return dst.length_squared() < radius * radius;
        }
        false
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        let map = TextureMapping2D::create(render_from_texture, parameters.parameter_dictionary())?;
        let tex1 =
            parameters.get_spectrum_texture_typed("tex1", &Spectrum::from(1.0), spectrum_type)?;
        let tex2 =
            parameters.get_spectrum_texture_typed("tex2", &Spectrum::from(0.0), spectrum_type)?;
        Ok(DotsSpectrumTexture::new(map, &tex1, &tex2))
    }
}
