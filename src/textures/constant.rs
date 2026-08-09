use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

pub struct ConstantTexture<T> {
    pub value: T,
}

impl<T: Clone> ConstantTexture<T> {
    pub fn new(value: &T) -> Self {
        ConstantTexture::<T> {
            value: value.clone(),
        }
    }

    pub fn evaluate(&self, _ctx: &TextureEvalContext) -> T {
        self.value.clone()
    }
}

impl ConstantTexture<Float> {
    pub fn create(
        _render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<FloatTexture, PbrtError> {
        let value = parameters.get_one_float("value", 1.0);
        Ok(FloatTexture::Constant(ConstantTexture::<Float>::new(
            &value,
        )))
    }
}

impl ConstantTexture<Spectrum> {
    pub fn create_from_params(
        _render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        let value = parameters.get_one_spectrum_typed("value", &Spectrum::one(), spectrum_type);
        Ok(ConstantTexture::<Spectrum>::new(&value))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RGBConstantSpectrumTexture {
    pub rgb: [Float; 3],
    pub spectrum_type: SpectrumType,
}

impl RGBConstantSpectrumTexture {
    pub fn new(rgb: &[Float; 3], spectrum_type: SpectrumType) -> Self {
        Self {
            rgb: *rgb,
            spectrum_type,
        }
    }

    pub fn evaluate(
        &self,
        _ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        Spectrum::rgb_to_sampled(self.rgb, self.spectrum_type, lambda)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampledCurveSpectrumTexture {
    pub lambda: Vec<Float>,
    pub values: Vec<Float>,
}

impl SampledCurveSpectrumTexture {
    pub fn new(lambda: &[Float], values: &[Float]) -> Self {
        Self {
            lambda: lambda.to_vec(),
            values: values.to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlackbodyConstantSpectrumTexture {
    pub values: Vec<Float>,
}

impl BlackbodyConstantSpectrumTexture {
    pub fn new(values: &[Float]) -> Self {
        Self {
            values: values.to_vec(),
        }
    }
}

pub fn spectrum_texture_from_rgb_constant(
    rgb: &[Float; 3],
    spectrum_type: SpectrumType,
) -> SpectrumTexture {
    SpectrumTexture::RGBConstant(RGBConstantSpectrumTexture::new(rgb, spectrum_type))
}

pub fn spectrum_texture_from_sampled_curve(lambda: &[Float], values: &[Float]) -> SpectrumTexture {
    SpectrumTexture::SampledCurve(SampledCurveSpectrumTexture::new(lambda, values))
}

pub fn spectrum_texture_from_blackbody(values: &[Float]) -> SpectrumTexture {
    SpectrumTexture::BlackbodyConstant(BlackbodyConstantSpectrumTexture::new(values))
}

pub fn spectrum_texture_from_constant(source: Spectrum) -> SpectrumTexture {
    SpectrumTexture::Constant(ConstantTexture::new(&source))
}

pub fn spectrum_texture_from_named_spectrum(name: &str) -> Option<SpectrumTexture> {
    spectrum_from_named(name).map(spectrum_texture_from_constant)
}
