use crate::util::base::Float;
use crate::util::misc::read_float_file;

use super::composite::Spectrum;
use super::config::{LAMBDA_MAX, LAMBDA_MIN};
use super::densely_sampled::DenselySampledSpectrum;
use super::helpers::interpolate_spectrum_samples;
use super::sampled::{SampledSpectrum, SampledWavelengths};

#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseLinearSpectrum {
    pub lambda: Vec<Float>,
    pub values: Vec<Float>,
}

impl PiecewiseLinearSpectrum {
    pub fn new(lambda: Vec<Float>, values: Vec<Float>) -> Self {
        assert_eq!(lambda.len(), values.len());
        for i in 0..lambda.len().saturating_sub(1) {
            assert!(lambda[i] < lambda[i + 1]);
        }
        Self { lambda, values }
    }

    pub fn read(path: &str) -> Option<Spectrum> {
        let vals = read_float_file(path).ok()?;
        if vals.is_empty() || vals.len() % 2 != 0 {
            return None;
        }

        let mut lambda = Vec::with_capacity(vals.len() / 2);
        let mut values = Vec::with_capacity(vals.len() / 2);
        for chunk in vals.chunks_exact(2) {
            if let Some(last_lambda) = lambda.last() {
                if chunk[0] <= *last_lambda {
                    return None;
                }
            }
            lambda.push(chunk[0]);
            values.push(chunk[1]);
        }

        Some(Spectrum::PiecewiseLinear(Self::new(lambda, values)))
    }

    pub fn from_interleaved(samples: &[Float], normalize: bool) -> Option<Self> {
        if samples.is_empty() || samples.len() % 2 != 0 {
            return None;
        }

        let n = samples.len() / 2;
        let mut lambda = Vec::with_capacity(n + 2);
        let mut values = Vec::with_capacity(n + 2);

        if samples[0] > LAMBDA_MIN as Float {
            lambda.push(LAMBDA_MIN as Float - 1.0);
            values.push(samples[1]);
        }

        for i in 0..n {
            let l = samples[2 * i];
            let v = samples[2 * i + 1];
            if i > 0 && l <= samples[2 * (i - 1)] {
                return None;
            }
            lambda.push(l);
            values.push(v);
        }

        if *lambda.last()? < LAMBDA_MAX as Float {
            lambda.push(LAMBDA_MAX as Float + 1.0);
            values.push(*values.last()?);
        }

        let mut spectrum = Self::new(lambda, values);
        if normalize {
            let y = spectrum.evaluate().y();
            if y > 0.0 {
                spectrum.scale(1.0 / y);
            }
        }

        Some(spectrum)
    }

    pub fn evaluate(&self) -> DenselySampledSpectrum {
        DenselySampledSpectrum::sample_function(|lambda| self.sample_at(lambda))
    }

    pub fn scale(&mut self, s: Float) {
        for value in &mut self.values {
            *value *= s;
        }
    }

    pub fn sample_at(&self, lambda: Float) -> Float {
        if self.lambda.is_empty()
            || lambda < self.lambda[0]
            || lambda > self.lambda[self.lambda.len() - 1]
        {
            return 0.0;
        }
        interpolate_spectrum_samples(&self.lambda, &self.values, lambda)
    }

    pub fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut values = [0.0; SampledSpectrum::N_SAMPLES];
        for i in 0..SampledSpectrum::N_SAMPLES {
            values[i] = self.sample_at(lambda[i]);
        }
        SampledSpectrum::from(values)
    }

    pub fn max_value(&self) -> Float {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().copied().fold(self.values[0], Float::max)
    }
}
