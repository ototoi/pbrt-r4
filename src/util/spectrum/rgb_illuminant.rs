use crate::util::base::Float;

use super::densely_sampled::DenselySampledSpectrum;
use super::rgb_to_spectrum::{RGBColorSpace, RGBColorSpaceIlluminant, RGBSigmoidPolynomial, SRGB};
use super::sampled::{SampledSpectrum, SampledWavelengths};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RGBIlluminantSpectrum {
    scale: Float,
    rsp: RGBSigmoidPolynomial,
    illuminant: RGBColorSpaceIlluminant,
}

impl RGBIlluminantSpectrum {
    pub fn new(rgb: [Float; 3]) -> Self {
        Self::from_color_space(&SRGB, rgb)
    }

    pub fn from_color_space(color_space: &RGBColorSpace, rgb: [Float; 3]) -> Self {
        let (scale, rsp) = color_space.illuminant_to_scaled_polynomial(rgb);
        Self {
            scale,
            rsp,
            illuminant: color_space.illuminant,
        }
    }

    pub fn evaluate(&self) -> DenselySampledSpectrum {
        DenselySampledSpectrum::sample_function(|lambda| self.sample_at(lambda))
    }

    pub fn sample_at(&self, lambda: Float) -> Float {
        self.scale * self.rsp.eval(lambda) * self.illuminant.sample_at(lambda)
    }

    pub fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut values = [0.0; SampledSpectrum::N_SAMPLES];
        for i in 0..SampledSpectrum::N_SAMPLES {
            values[i] = self.scale * self.rsp.eval(lambda[i]);
        }
        SampledSpectrum::from(values) * self.illuminant.sample(lambda)
    }

    pub fn max_value(&self) -> Float {
        self.scale * self.rsp.max_value() * self.illuminant.max_value()
    }

    /// Densely sampled view of the underlying illuminant only
    /// (no RGB scale, no reflectance polynomial). Mirrors pbrt-v4
    /// `RGBIlluminantSpectrum::Illuminant()` and is used by
    /// `spectrum_to_photometric` so the photometric integral stays
    /// invariant to the RGB magnitude.
    pub fn illuminant_dense(&self) -> DenselySampledSpectrum {
        self.illuminant.to_dense()
    }
}
