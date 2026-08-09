use crate::util::base::Float;

use super::densely_sampled::DenselySampledSpectrum;
use super::rgb_to_spectrum::{srgb_albedo_to_polynomial, RGBColorSpace, RGBSigmoidPolynomial};
use super::sampled::{SampledSpectrum, SampledWavelengths};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RGBAlbedoSpectrum {
    rsp: RGBSigmoidPolynomial,
}

impl RGBAlbedoSpectrum {
    pub fn new(rgb: [Float; 3]) -> Self {
        Self {
            rsp: srgb_albedo_to_polynomial(rgb),
        }
    }

    pub fn from_color_space(color_space: &RGBColorSpace, rgb: [Float; 3]) -> Self {
        Self {
            rsp: color_space.albedo_to_polynomial(rgb),
        }
    }

    pub fn evaluate(&self) -> DenselySampledSpectrum {
        DenselySampledSpectrum::sample_function(|lambda| self.sample_at(lambda))
    }

    pub fn sample_at(&self, lambda: Float) -> Float {
        self.rsp.eval(lambda)
    }

    pub fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut values = [0.0; SampledSpectrum::N_SAMPLES];
        for i in 0..SampledSpectrum::N_SAMPLES {
            values[i] = self.rsp.eval(lambda[i]);
        }
        SampledSpectrum::from(values)
    }

    pub fn max_value(&self) -> Float {
        self.rsp.max_value()
    }
}
