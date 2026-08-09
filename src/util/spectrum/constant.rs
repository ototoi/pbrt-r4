use crate::util::base::Float;

use super::densely_sampled::DenselySampledSpectrum;
use super::sampled::{SampledSpectrum, SampledWavelengths};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantSpectrum {
    pub value: Float,
}

impl ConstantSpectrum {
    pub fn new(value: Float) -> Self {
        Self { value }
    }

    pub fn evaluate(&self) -> DenselySampledSpectrum {
        DenselySampledSpectrum::from(self.value)
    }

    pub fn sample_at(&self, _lambda: Float) -> Float {
        self.value
    }

    pub fn sample(&self, _lambda: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::new(self.value)
    }

    pub fn max_value(&self) -> Float {
        self.value
    }
}
