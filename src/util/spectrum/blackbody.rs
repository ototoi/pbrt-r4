use crate::util::base::Float;

use super::densely_sampled::DenselySampledSpectrum;

fn blackbody_scalar(lambda_nm: f64, temperature: f64) -> f64 {
    if temperature <= 0.0 {
        return 0.0;
    }

    const C: f64 = 299_792_458.0;
    const H: f64 = 6.626_069_57e-34;
    const KB: f64 = 1.380_648_8e-23;

    let lambda = lambda_nm * 1e-9;
    let lambda5 = (lambda * lambda) * (lambda * lambda) * lambda;
    (2.0 * H * C * C) / (lambda5 * (f64::exp((H * C) / (lambda * KB * temperature)) - 1.0))
}

pub fn blackbody(lambda: &[f64], temperature: f64) -> Vec<f64> {
    lambda
        .iter()
        .map(|wavelength| blackbody_scalar(*wavelength, temperature))
        .collect()
}

pub fn blackbody_normalized_at(lambda: Float, temperature: Float) -> Float {
    if temperature <= 0.0 {
        return 0.0;
    }

    let temperature = temperature as f64;
    let value = blackbody_scalar(lambda as f64, temperature);
    let lambda_max_nm = 2.897_772_1e-3 / temperature * 1e9;
    let peak = blackbody_scalar(lambda_max_nm, temperature);
    (value / peak) as Float
}

pub fn blackbody_normalized(lambda: &[Float], temperature: Float) -> Vec<Float> {
    lambda
        .iter()
        .map(|wavelength| blackbody_normalized_at(*wavelength, temperature))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlackbodySpectrum {
    pub temperature: Float,
    pub scale: Float,
    normalization_factor: Float,
}

impl BlackbodySpectrum {
    pub fn new(temperature: Float, scale: Float) -> Self {
        let normalization_factor = if temperature > 0.0 {
            let lambda_max_nm = 2.897_772_1e-3 / temperature as f64 * 1e9;
            let peak = blackbody_scalar(lambda_max_nm, temperature as f64) as Float;
            if peak > 0.0 {
                scale / peak
            } else {
                0.0
            }
        } else {
            0.0
        };
        Self {
            temperature,
            scale,
            normalization_factor,
        }
    }

    pub fn evaluate(&self) -> DenselySampledSpectrum {
        DenselySampledSpectrum::sample_function(|lambda| self.sample_at(lambda))
    }

    pub fn sample_at(&self, lambda: Float) -> Float {
        (blackbody_scalar(lambda as f64, self.temperature as f64) as Float)
            * self.normalization_factor
    }

    pub fn sample(
        &self,
        lambda: &super::sampled::SampledWavelengths,
    ) -> super::sampled::SampledSpectrum {
        let mut values = [0.0; super::sampled::SampledSpectrum::N_SAMPLES];
        for i in 0..super::sampled::SampledSpectrum::N_SAMPLES {
            values[i] = self.sample_at(lambda[i]);
        }
        super::sampled::SampledSpectrum::from(values)
    }

    pub fn max_value(&self) -> Float {
        self.scale.max(0.0)
    }
}
