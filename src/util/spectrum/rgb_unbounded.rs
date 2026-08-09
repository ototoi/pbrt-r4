use crate::util::base::Float;

use super::densely_sampled::DenselySampledSpectrum;
use super::rgb_to_spectrum::{
    srgb_unbounded_to_scaled_polynomial, RGBColorSpace, RGBSigmoidPolynomial,
};
use super::sampled::{SampledSpectrum, SampledWavelengths};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RGBUnboundedSpectrum {
    scale: Float,
    rsp: RGBSigmoidPolynomial,
}

impl RGBUnboundedSpectrum {
    pub fn new(rgb: [Float; 3]) -> Self {
        let (scale, rsp) = srgb_unbounded_to_scaled_polynomial(rgb);
        Self { scale, rsp }
    }

    pub fn from_color_space(color_space: &RGBColorSpace, rgb: [Float; 3]) -> Self {
        let (scale, rsp) = color_space.unbounded_to_scaled_polynomial(rgb);
        Self { scale, rsp }
    }

    pub fn evaluate(&self) -> DenselySampledSpectrum {
        DenselySampledSpectrum::sample_function(|lambda| self.sample_at(lambda))
    }

    pub fn sample_at(&self, lambda: Float) -> Float {
        self.scale * self.rsp.eval(lambda)
    }

    pub fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut values = [0.0; SampledSpectrum::N_SAMPLES];
        for i in 0..SampledSpectrum::N_SAMPLES {
            values[i] = self.scale * self.rsp.eval(lambda[i]);
        }
        SampledSpectrum::from(values)
    }

    pub fn max_value(&self) -> Float {
        self.scale * self.rsp.max_value()
    }
}
