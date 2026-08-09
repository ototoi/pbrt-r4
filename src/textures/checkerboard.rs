use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

use std::sync::Arc;

// AAMethod Declaration
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AAMethod {
    None,
    ClosedForm,
}

/// Result of classifying a shading point against the checkerboard
/// pattern: pick one sub-texture, the other, or blend by `area2` (the
/// fraction covered by `tex2`). Shared by `Spectrum`/`SampledSpectrum`
/// evaluation paths so the geometry-only decision is computed once.
enum CheckerboardChoice {
    Tex1,
    Tex2,
    Blend { area2: Float },
}

// 2D Float variant
pub struct Checkerboard2DTexture<T> {
    mapping: TextureMapping2D,
    tex1: Arc<FloatTexture>,
    tex2: Arc<FloatTexture>,
    aa_method: AAMethod,
    _phantom: std::marker::PhantomData<T>,
}

impl Checkerboard2DTexture<Float> {
    pub fn new(
        mapping: TextureMapping2D,
        tex1: &Arc<FloatTexture>,
        tex2: &Arc<FloatTexture>,
        aa_method: AAMethod,
    ) -> Self {
        return Checkerboard2DTexture::<Float> {
            mapping,
            tex1: Arc::clone(tex1),
            tex2: Arc::clone(tex2),
            aa_method,
            _phantom: std::marker::PhantomData,
        };
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let (st, dstdx, dstdy) = self.mapping.map(ctx);
        if self.aa_method == AAMethod::None {
            if ((Float::floor(st[0]) as i32) + (Float::floor(st[1]) as i32)) % 2 == 0 {
                return self.tex1.evaluate(ctx);
            } else {
                return self.tex2.evaluate(ctx);
            }
        } else {
            let ds = Float::max(Float::abs(dstdx[0]), Float::abs(dstdy[0]));
            let dt = Float::max(Float::abs(dstdx[1]), Float::abs(dstdy[1]));
            let s0 = st[0] - ds;
            let s1 = st[0] + ds;
            let t0 = st[1] - dt;
            let t1 = st[1] + dt;
            if Float::floor(s0) == Float::floor(s1) && Float::floor(t0) == Float::floor(t1) {
                if ((Float::floor(st[0]) as i32) + (Float::floor(st[1]) as i32)) % 2 == 0 {
                    return self.tex1.evaluate(ctx);
                } else {
                    return self.tex2.evaluate(ctx);
                }
            }

            let sint = (bump_int(s1) - bump_int(s0)) / (2.0 * ds);
            let tint = (bump_int(t1) - bump_int(t0)) / (2.0 * dt);
            let mut area2 = sint + tint - 2.0 * sint * tint;
            if ds > 1.0 || dt > 1.0 {
                area2 = 0.5;
            }
            return self.tex1.evaluate(ctx) * (1.0 - area2) + self.tex2.evaluate(ctx) * area2;
        }
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<FloatTexture, PbrtError> {
        let dim = parameters.get_one_int("dimension", 2);
        if dim != 2 && dim != 3 {
            return Err(PbrtError::error(&format!(
                "{} dimensional checkerboard texture not supported",
                dim
            )));
        }
        let tex1 = parameters.get_float_texture("tex1", 1.0)?;
        let tex2 = parameters.get_float_texture("tex2", 0.0)?;
        if dim == 2 {
            let map =
                TextureMapping2D::create(render_from_texture, parameters.parameter_dictionary())?;
            let aa_method_str = parameters.get_one_string("aamode", "closedform");
            let aa_method = if aa_method_str == "none" {
                AAMethod::None
            } else if aa_method_str == "closedform" {
                AAMethod::ClosedForm
            } else {
                return Err(PbrtError::error(&format!(
                    "Checkerboard aamode \"{}\" unknown",
                    aa_method_str
                )));
            };
            Ok(FloatTexture::Checkerboard2D(Checkerboard2DTexture::new(
                map, &tex1, &tex2, aa_method,
            )))
        } else {
            let map =
                TextureMapping3D::create(parameters.parameter_dictionary(), render_from_texture);
            Ok(FloatTexture::Checkerboard3D(Checkerboard3DTexture::new(
                map, &tex1, &tex2,
            )))
        }
    }
}

#[inline]
fn bump_int(x: Float) -> Float {
    return Float::floor(x / 2.0) + 2.0 * Float::max(x / 2.0 - Float::floor(x / 2.0) - 0.5, 0.0);
}

// 2D Spectrum variant
pub struct Checkerboard2DSpectrumTexture {
    mapping: TextureMapping2D,
    tex1: Arc<SpectrumTexture>,
    tex2: Arc<SpectrumTexture>,
    aa_method: AAMethod,
}

impl Checkerboard2DSpectrumTexture {
    pub fn new(
        mapping: TextureMapping2D,
        tex1: &Arc<SpectrumTexture>,
        tex2: &Arc<SpectrumTexture>,
        aa_method: AAMethod,
    ) -> Self {
        return Checkerboard2DSpectrumTexture {
            mapping,
            tex1: Arc::clone(tex1),
            tex2: Arc::clone(tex2),
            aa_method,
        };
    }

    pub fn evaluate(
        &self,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        match self.classify(ctx) {
            CheckerboardChoice::Tex1 => self.tex1.evaluate(ctx, lambda),
            CheckerboardChoice::Tex2 => self.tex2.evaluate(ctx, lambda),
            CheckerboardChoice::Blend { area2 } => {
                self.tex1.evaluate(ctx, lambda) * (1.0 - area2)
                    + self.tex2.evaluate(ctx, lambda) * area2
            }
        }
    }

    fn classify(&self, ctx: &TextureEvalContext) -> CheckerboardChoice {
        let (st, dstdx, dstdy) = self.mapping.map(ctx);
        let pick = |st: Point2f| -> CheckerboardChoice {
            if ((Float::floor(st[0]) as i32) + (Float::floor(st[1]) as i32)) % 2 == 0 {
                CheckerboardChoice::Tex1
            } else {
                CheckerboardChoice::Tex2
            }
        };
        if self.aa_method == AAMethod::None {
            return pick(st);
        }
        let ds = Float::max(Float::abs(dstdx[0]), Float::abs(dstdy[0]));
        let dt = Float::max(Float::abs(dstdx[1]), Float::abs(dstdy[1]));
        let s0 = st[0] - ds;
        let s1 = st[0] + ds;
        let t0 = st[1] - dt;
        let t1 = st[1] + dt;
        if Float::floor(s0) == Float::floor(s1) && Float::floor(t0) == Float::floor(t1) {
            return pick(st);
        }
        let sint = (bump_int(s1) - bump_int(s0)) / (2.0 * ds);
        let tint = (bump_int(t1) - bump_int(t0)) / (2.0 * dt);
        let mut area2 = sint + tint - 2.0 * sint * tint;
        if ds > 1.0 || dt > 1.0 {
            area2 = 0.5;
        }
        CheckerboardChoice::Blend { area2 }
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        let tex1 =
            parameters.get_spectrum_texture_typed("tex1", &Spectrum::from(1.0), spectrum_type)?;
        let tex2 =
            parameters.get_spectrum_texture_typed("tex2", &Spectrum::from(0.0), spectrum_type)?;
        let map = TextureMapping2D::create(render_from_texture, parameters.parameter_dictionary())?;
        let aa_method_str = parameters.get_one_string("aamode", "closedform");
        let aa_method = if aa_method_str == "none" {
            AAMethod::None
        } else {
            AAMethod::ClosedForm
        };
        Ok(Checkerboard2DSpectrumTexture::new(
            map, &tex1, &tex2, aa_method,
        ))
    }
}

// 3D Float variant
pub struct Checkerboard3DTexture<T> {
    mapping: TextureMapping3D,
    tex1: Arc<FloatTexture>,
    tex2: Arc<FloatTexture>,
    _phantom: std::marker::PhantomData<T>,
}

impl Checkerboard3DTexture<Float> {
    pub fn new(
        mapping: TextureMapping3D,
        tex1: &Arc<FloatTexture>,
        tex2: &Arc<FloatTexture>,
    ) -> Self {
        return Checkerboard3DTexture::<Float> {
            mapping,
            tex1: Arc::clone(tex1),
            tex2: Arc::clone(tex2),
            _phantom: std::marker::PhantomData,
        };
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let (st, _dstdx, _dstdy) = self.mapping.map(ctx);
        if ((Float::floor(st[0]) as i32)
            + (Float::floor(st[1]) as i32)
            + (Float::floor(st[2]) as i32))
            % 2
            == 0
        {
            return self.tex1.evaluate(ctx);
        } else {
            return self.tex2.evaluate(ctx);
        }
    }
}

// 3D Spectrum variant
pub struct Checkerboard3DSpectrumTexture {
    mapping: TextureMapping3D,
    tex1: Arc<SpectrumTexture>,
    tex2: Arc<SpectrumTexture>,
}

impl Checkerboard3DSpectrumTexture {
    pub fn new(
        mapping: TextureMapping3D,
        tex1: &Arc<SpectrumTexture>,
        tex2: &Arc<SpectrumTexture>,
    ) -> Self {
        return Checkerboard3DSpectrumTexture {
            mapping,
            tex1: Arc::clone(tex1),
            tex2: Arc::clone(tex2),
        };
    }

    pub fn evaluate(
        &self,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        if self.pick_tex1(ctx) {
            self.tex1.evaluate(ctx, lambda)
        } else {
            self.tex2.evaluate(ctx, lambda)
        }
    }

    fn pick_tex1(&self, ctx: &TextureEvalContext) -> bool {
        let (st, _dstdx, _dstdy) = self.mapping.map(ctx);
        ((Float::floor(st[0]) as i32) + (Float::floor(st[1]) as i32) + (Float::floor(st[2]) as i32))
            % 2
            == 0
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        let tex1 =
            parameters.get_spectrum_texture_typed("tex1", &Spectrum::from(1.0), spectrum_type)?;
        let tex2 =
            parameters.get_spectrum_texture_typed("tex2", &Spectrum::from(0.0), spectrum_type)?;
        let map = TextureMapping3D::create(parameters.parameter_dictionary(), render_from_texture);
        Ok(Checkerboard3DSpectrumTexture::new(map, &tex1, &tex2))
    }
}
