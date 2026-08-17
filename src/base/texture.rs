// Texture enum - base interface for all textures
// Changed from trait to enum-based approach

use crate::paramdict::*;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::imageio::ColorEncoding;
use crate::util::spectrum::*;
use crate::util::transform::*;

use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImageWrap {
    Repeat,
    Black,
    Clamp,
    OctahedralSphere,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImageFilter {
    Point,
    Bilinear,
    Trilinear,
    EWA,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TexInfo {
    pub cache_version: u32,
    pub filename: String,
    pub filter: ImageFilter,
    pub max_aniso: Float,
    pub swrap_mode: ImageWrap,
    pub twrap_mode: ImageWrap,
    pub scale: Float,
    pub encoding: ColorEncoding,
    pub flip_y: bool,
}

impl Display for TexInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_string(&self).unwrap();
        return write!(f, "{}", s);
    }
}

pub fn typed_spectrum_texture_name(name: &str, spectrum_type: SpectrumType) -> String {
    format!("{}::{}", spectrum_type.as_str(), name)
}

// Float Texture Enum
pub enum FloatTexture {
    Constant(ConstantTexture<Float>),
    Scale(ScaleTexture<Float, Float>),
    Mix(MixTexture<Float>),
    DirectionMix(DirectionMixFloatTexture),
    Bilerp(BilerpTexture<Float>),
    ImageMap(ImageTexture<Float, Float>),
    Checkerboard2D(Checkerboard2DTexture<Float>),
    Checkerboard3D(Checkerboard3DTexture<Float>),
    Dots(DotsTexture<Float>),
    FBm(FBmTexture),
    Wrinkled(WrinkledTexture),
    Windy(WindyTexture),
}

impl FloatTexture {
    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        use FloatTexture::*;
        match self {
            Constant(tex) => tex.evaluate(ctx),
            Scale(tex) => tex.evaluate(ctx),
            Mix(tex) => tex.evaluate(ctx),
            DirectionMix(tex) => tex.evaluate(ctx),
            Bilerp(tex) => tex.evaluate(ctx),
            ImageMap(tex) => tex.evaluate(ctx),
            Checkerboard2D(tex) => tex.evaluate(ctx),
            Checkerboard3D(tex) => tex.evaluate(ctx),
            Dots(tex) => tex.evaluate(ctx),
            FBm(tex) => tex.evaluate(ctx),
            Wrinkled(tex) => tex.evaluate(ctx),
            Windy(tex) => tex.evaluate(ctx),
        }
    }

    pub fn create(
        tex_type: &str,
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<FloatTexture, PbrtError> {
        match tex_type {
            "constant" => ConstantTexture::<Float>::create(render_from_texture, parameters),
            "scale" => ScaleTexture::<Float, Float>::create(render_from_texture, parameters),
            "mix" => MixTexture::<Float>::create(render_from_texture, parameters),
            "directionmix" => Ok(FloatTexture::DirectionMix(
                DirectionMixFloatTexture::create(render_from_texture, parameters)?,
            )),
            "bilerp" => BilerpTexture::<Float>::create(render_from_texture, parameters),
            "imagemap" => FloatImageTexture::create(render_from_texture, parameters),
            "checkerboard" => {
                Checkerboard2DTexture::<Float>::create(render_from_texture, parameters)
            }
            "dots" => DotsTexture::<Float>::create(render_from_texture, parameters),
            "fbm" => Ok(FloatTexture::FBm(FBmTexture::create(
                render_from_texture,
                parameters,
            )?)),
            "wrinkled" => Ok(FloatTexture::Wrinkled(WrinkledTexture::create(
                render_from_texture,
                parameters,
            )?)),
            "windy" => Ok(FloatTexture::Windy(WindyTexture::create(
                render_from_texture,
                parameters,
            )?)),
            _ => Err(PbrtError::error(&format!(
                "Float texture type '{}' unknown",
                tex_type
            ))),
        }
    }
}

// Spectrum Texture Enum
pub enum SpectrumTexture {
    Constant(ConstantTexture<Spectrum>),
    RGBConstant(RGBConstantSpectrumTexture),
    SampledCurve(SampledCurveSpectrumTexture),
    BlackbodyConstant(BlackbodyConstantSpectrumTexture),
    Scale(ScaleSpectrumTexture),
    Mix(MixSpectrumTexture),
    DirectionMix(DirectionMixSpectrumTexture),
    Bilerp(BilerpTexture<Spectrum>),
    ImageMap(ImageTexture<RGBSpectrum, Spectrum>),
    Checkerboard2D(Checkerboard2DSpectrumTexture),
    Checkerboard3D(Checkerboard3DSpectrumTexture),
    Dots(DotsSpectrumTexture),
    FBm(FBmTexture),
    Wrinkled(WrinkledTexture),
    Marble(MarbleTexture),
    Windy(WindyTexture),
    UV(UVTexture),
    Normal(NormalTexture),
}

impl SpectrumTexture {
    /// pbrt-v4 `texEval(SpectrumTexture, ctx, lambda)` (textures.h) —
    /// every variant has a `SampledSpectrum`-native path. The cold
    /// procedural / Spectrum-returning variants (Bilerp/Marble/UV/
    /// Normal/FBm/Wrinkled/Windy/SampledCurve/BlackbodyConstant) build
    /// a Spectrum at this site and sample it; the hot ones
    /// (ImageMap/RGBConstant/Scale/Mix/DirectionMix/Checkerboard*/Dots)
    /// stay in SampledSpectrum throughout.
    pub fn evaluate(
        &self,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        use SpectrumTexture::*;
        match self {
            Constant(tex) => tex.value.sample(lambda),
            RGBConstant(tex) => tex.evaluate(ctx, lambda),
            ImageMap(tex) => tex.evaluate(ctx, lambda),
            Scale(tex) => tex.evaluate(ctx, lambda),
            Mix(tex) => tex.evaluate(ctx, lambda),
            DirectionMix(tex) => tex.evaluate(ctx, lambda),
            Checkerboard2D(tex) => tex.evaluate(ctx, lambda),
            Checkerboard3D(tex) => tex.evaluate(ctx, lambda),
            Dots(tex) => tex.evaluate(ctx, lambda),
            SampledCurve(tex) => Spectrum::from_sampled(&tex.lambda, &tex.values).sample(lambda),
            BlackbodyConstant(tex) => {
                Spectrum::from(&DenseSampledSpectrum::from_blackbody(&tex.values)).sample(lambda)
            }
            Bilerp(tex) => tex.evaluate(ctx).sample(lambda),
            Marble(tex) => tex.evaluate(ctx).sample(lambda),
            UV(tex) => tex.evaluate(ctx).sample(lambda),
            Normal(tex) => tex.evaluate(ctx).sample(lambda),
            FBm(tex) => SampledSpectrum::new(tex.evaluate(ctx)),
            Wrinkled(tex) => SampledSpectrum::new(tex.evaluate(ctx)),
            Windy(tex) => SampledSpectrum::new(tex.evaluate(ctx)),
        }
    }

    pub fn create(
        tex_type: &str,
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
    ) -> Result<SpectrumTexture, PbrtError> {
        match tex_type {
            "constant" => {
                if let Some(source) = parameters.get_spectrum_or_null_typed("value", spectrum_type)
                {
                    Ok(spectrum_texture_from_constant(source))
                } else {
                    Ok(SpectrumTexture::Constant(
                        ConstantTexture::<Spectrum>::create_from_params(
                            render_from_texture,
                            parameters,
                            spectrum_type,
                        )?,
                    ))
                }
            }
            "scale" => Ok(SpectrumTexture::Scale(ScaleSpectrumTexture::create(
                render_from_texture,
                parameters,
                spectrum_type,
            )?)),
            "mix" => Ok(SpectrumTexture::Mix(MixSpectrumTexture::create(
                render_from_texture,
                parameters,
                spectrum_type,
            )?)),
            "directionmix" => Ok(SpectrumTexture::DirectionMix(
                DirectionMixSpectrumTexture::create(
                    render_from_texture,
                    parameters,
                    spectrum_type,
                )?,
            )),
            "bilerp" => Ok(SpectrumTexture::Bilerp(BilerpTexture::<Spectrum>::create(
                render_from_texture,
                parameters,
                spectrum_type,
            )?)),
            "imagemap" => Ok(SpectrumTexture::ImageMap(SpectrumImageTexture::create(
                render_from_texture,
                parameters,
                spectrum_type,
            )?)),
            "checkerboard" => {
                let dim = parameters.get_one_int("dimension", 2);
                match dim {
                    2 => Ok(SpectrumTexture::Checkerboard2D(
                        Checkerboard2DSpectrumTexture::create(
                            render_from_texture,
                            parameters,
                            spectrum_type,
                        )?,
                    )),
                    3 => Ok(SpectrumTexture::Checkerboard3D(
                        Checkerboard3DSpectrumTexture::create(
                            render_from_texture,
                            parameters,
                            spectrum_type,
                        )?,
                    )),
                    _ => Err(PbrtError::error(&format!(
                        "{} dimensional checkerboard texture not supported",
                        dim
                    ))),
                }
            }
            "dots" => Ok(SpectrumTexture::Dots(DotsSpectrumTexture::create(
                render_from_texture,
                parameters,
                spectrum_type,
            )?)),
            "fbm" => Ok(SpectrumTexture::FBm(FBmTexture::create(
                render_from_texture,
                parameters,
            )?)),
            "wrinkled" => Ok(SpectrumTexture::Wrinkled(WrinkledTexture::create(
                render_from_texture,
                parameters,
            )?)),
            "marble" => Ok(SpectrumTexture::Marble(MarbleTexture::create(
                render_from_texture,
                parameters,
                spectrum_type,
            )?)),
            "windy" => Ok(SpectrumTexture::Windy(WindyTexture::create(
                render_from_texture,
                parameters,
            )?)),
            "uv" => Ok(SpectrumTexture::UV(UVTexture::create(
                render_from_texture,
                parameters,
                spectrum_type,
            )?)),
            "normal" => Ok(SpectrumTexture::Normal(NormalTexture::create(
                render_from_texture,
                parameters,
                spectrum_type,
            )?)),
            _ => Err(PbrtError::error(&format!(
                "Spectrum texture type '{}' unknown",
                tex_type
            ))),
        }
    }
}
