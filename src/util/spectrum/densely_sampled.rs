use crate::util::base::Float;
use crate::util::error::PbrtError;
use crate::util::misc::read_float_file;
use std::sync::Arc;

use super::blackbody::blackbody_normalized_at;
use super::cie::{xyz_to_rgb, CIE_SAMPLES, CIE_X, CIE_Y, CIE_Y_INTEGRAL, CIE_Z};
use super::composite::Spectrum;
use super::config::{LAMBDA_MAX, LAMBDA_MIN};
use super::helpers::{
    interpolate_spectrum_samples, lerp, sort_spectrum_samples, spectrum_samples_sorted,
};
use super::rgb_to_spectrum::{
    srgb_albedo_to_dense_spectrum, srgb_illuminant_to_dense_spectrum,
    srgb_unbounded_to_dense_spectrum,
};
use super::sampled::{SampledSpectrum, SampledWavelengths};

pub const DENSE_SPECTRUM_SAMPLES: usize = (LAMBDA_MAX - LAMBDA_MIN + 1) as usize;

#[derive(Debug, Clone, PartialEq)]
pub struct DenselySampledSpectrum {
    values: Arc<[Float; DENSE_SPECTRUM_SAMPLES]>,
}

pub type DenseSampledSpectrum = DenselySampledSpectrum;

impl DenselySampledSpectrum {
    pub const N_SAMPLES: usize = DENSE_SPECTRUM_SAMPLES;

    pub fn new(value: Float) -> Self {
        Self {
            values: Arc::new([value; DENSE_SPECTRUM_SAMPLES]),
        }
    }

    pub fn zero() -> Self {
        Self::new(0.0)
    }

    pub fn one() -> Self {
        Self::new(1.0)
    }

    #[inline]
    fn lambda_at(index: usize) -> Float {
        LAMBDA_MIN as Float + index as Float
    }

    pub fn sample_function<F>(f: F) -> Self
    where
        F: Fn(Float) -> Float,
    {
        let mut dense = [0.0; DENSE_SPECTRUM_SAMPLES];
        for (i, value) in dense.iter_mut().enumerate() {
            *value = f(Self::lambda_at(i));
        }
        Self {
            values: Arc::new(dense),
        }
    }

    /// Build directly from a slice of `DENSE_SPECTRUM_SAMPLES`
    /// 1nm samples. Used for spectra precomputed at build time
    /// (e.g., the canon sensor curves emitted by
    /// `build/sensor/build_pixel_sensor.rs`).
    pub fn from_dense_array(values: &[Float]) -> Self {
        assert_eq!(values.len(), DENSE_SPECTRUM_SAMPLES);
        let mut dense = [0.0; DENSE_SPECTRUM_SAMPLES];
        dense.copy_from_slice(values);
        Self {
            values: Arc::new(dense),
        }
    }

    pub fn from_sampled(lambda: &[Float], values: &[Float]) -> Self {
        if lambda.is_empty() || values.is_empty() {
            return Self::zero();
        }
        assert_eq!(lambda.len(), values.len());

        let (lambda, values) = if spectrum_samples_sorted(lambda, values) {
            (lambda.to_vec(), values.to_vec())
        } else {
            let mut sorted_lambda = lambda.to_vec();
            let mut sorted_values = values.to_vec();
            sort_spectrum_samples(&mut sorted_lambda, &mut sorted_values);
            (sorted_lambda, sorted_values)
        };

        let mut dense = [0.0; DENSE_SPECTRUM_SAMPLES];
        for (i, value) in dense.iter_mut().enumerate() {
            *value = interpolate_spectrum_samples(&lambda, &values, Self::lambda_at(i));
        }
        Self {
            values: Arc::new(dense),
        }
    }

    pub fn from_blackbody(values: &[Float]) -> Self {
        let mut dense = Self::zero();
        let mut i = 0;
        while i < values.len() {
            let temperature = values[i];
            let scale = if i + 1 < values.len() {
                values[i + 1]
            } else {
                1.0
            };
            if temperature > 0.0 {
                let dense_values = Arc::make_mut(&mut dense.values);
                for j in 0..DENSE_SPECTRUM_SAMPLES {
                    let lambda = Self::lambda_at(j);
                    dense_values[j] += scale * blackbody_normalized_at(lambda, temperature);
                }
            }
            i += 2;
        }
        dense
    }

    pub fn from_spectrum(spectrum: &Spectrum) -> Self {
        match spectrum {
            Spectrum::DenselySampled(dense) => dense.clone(),
            _ => Self::sample_function(|lambda| spectrum.sample_at(lambda)),
        }
    }

    pub fn from_rgb_reflectance(rgb: [Float; 3]) -> Self {
        srgb_albedo_to_dense_spectrum(rgb)
    }

    pub fn from_rgb_unbounded(rgb: [Float; 3]) -> Self {
        srgb_unbounded_to_dense_spectrum(rgb)
    }

    pub fn from_rgb_illuminant(rgb: [Float; 3]) -> Self {
        srgb_illuminant_to_dense_spectrum(rgb)
    }

    pub fn load_sampled_spectrum_file(path: &str) -> Result<Self, PbrtError> {
        let values = read_float_file(path)?;
        let pair_count = values.len() / 2;
        let mut lambda = Vec::with_capacity(pair_count);
        let mut samples = Vec::with_capacity(pair_count);
        for i in 0..pair_count {
            lambda.push(values[2 * i]);
            samples.push(values[2 * i + 1]);
        }
        Ok(Self::from_sampled(&lambda, &samples))
    }

    pub fn sample_at(&self, lambda: Float) -> Float {
        let lambda_min = LAMBDA_MIN as Float;
        let lambda_max = LAMBDA_MAX as Float;
        if lambda < lambda_min || lambda > lambda_max {
            return 0.0;
        }
        let offset = (lambda - lambda_min) / (lambda_max - lambda_min)
            * (DENSE_SPECTRUM_SAMPLES - 1) as Float;
        let index0 = offset.floor() as usize;
        let index1 = usize::min(index0 + 1, DENSE_SPECTRUM_SAMPLES - 1);
        let t = offset - index0 as Float;
        lerp(t, self.values[index0], self.values[index1])
    }

    pub fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut values = [0.0; SampledSpectrum::N_SAMPLES];
        for i in 0..SampledSpectrum::N_SAMPLES {
            values[i] = self.sample_at(lambda[i]);
        }
        SampledSpectrum::from(values)
    }

    pub fn clamp(&self, low: Float, high: Float) -> Self {
        let mut values = *self.values;
        for value in &mut values {
            *value = value.clamp(low, high);
        }
        Self {
            values: Arc::new(values),
        }
    }

    pub fn clamp_zero(&self) -> Self {
        self.clamp(0.0, Float::INFINITY)
    }

    pub fn sqrt(&self) -> Self {
        let mut values = *self.values;
        for value in &mut values {
            *value = value.sqrt();
        }
        Self {
            values: Arc::new(values),
        }
    }

    pub fn exp(&self) -> Self {
        let mut values = *self.values;
        for value in &mut values {
            *value = value.exp();
        }
        Self {
            values: Arc::new(values),
        }
    }

    pub fn average(&self) -> Float {
        self.values.iter().sum::<Float>() / DENSE_SPECTRUM_SAMPLES as Float
    }

    pub fn is_valid(&self) -> bool {
        self.values.iter().all(|value| value.is_finite())
    }

    pub fn scale(&mut self, s: Float) {
        for value in Arc::make_mut(&mut self.values).iter_mut() {
            *value *= s;
        }
    }

    pub fn max_component_value(&self) -> Float {
        let mut max_value = self.values[0];
        for i in 1..DENSE_SPECTRUM_SAMPLES {
            max_value = max_value.max(self.values[i]);
        }
        max_value
    }

    pub fn max_value(&self) -> Float {
        self.max_component_value()
    }

    pub fn y(&self) -> Float {
        self.to_xyz()[1]
    }

    pub fn to_xyz(&self) -> [Float; 3] {
        debug_assert_eq!(CIE_SAMPLES, DENSE_SPECTRUM_SAMPLES);

        let mut xyz = [0.0; 3];
        for i in 0..DENSE_SPECTRUM_SAMPLES {
            xyz[0] += CIE_X[i] * self.values[i];
            xyz[1] += CIE_Y[i] * self.values[i];
            xyz[2] += CIE_Z[i] * self.values[i];
        }

        let scale = (LAMBDA_MAX - LAMBDA_MIN) as Float
            / (CIE_Y_INTEGRAL * (DENSE_SPECTRUM_SAMPLES - 1) as Float);
        xyz[0] *= scale;
        xyz[1] *= scale;
        xyz[2] *= scale;
        xyz
    }

    pub fn to_rgb(&self) -> [Float; 3] {
        xyz_to_rgb(&self.to_xyz())
    }

    pub fn near_equal(a: &Self, b: &Self, eps: Float) -> bool {
        for i in 0..DENSE_SPECTRUM_SAMPLES {
            if (a.values[i] - b.values[i]).abs() > eps {
                return false;
            }
        }
        true
    }
}

impl Default for DenselySampledSpectrum {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<Float> for DenselySampledSpectrum {
    fn from(value: Float) -> Self {
        Self::new(value)
    }
}

impl std::ops::Index<usize> for DenselySampledSpectrum {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl std::ops::IndexMut<usize> for DenselySampledSpectrum {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut Arc::make_mut(&mut self.values)[index]
    }
}

impl std::ops::AddAssign for DenselySampledSpectrum {
    fn add_assign(&mut self, rhs: Self) {
        let values = Arc::make_mut(&mut self.values);
        for i in 0..DENSE_SPECTRUM_SAMPLES {
            values[i] += rhs.values[i];
        }
    }
}

impl std::ops::Add for DenselySampledSpectrum {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl std::ops::SubAssign for DenselySampledSpectrum {
    fn sub_assign(&mut self, rhs: Self) {
        let values = Arc::make_mut(&mut self.values);
        for i in 0..DENSE_SPECTRUM_SAMPLES {
            values[i] -= rhs.values[i];
        }
    }
}

impl std::ops::Sub for DenselySampledSpectrum {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl std::ops::MulAssign for DenselySampledSpectrum {
    fn mul_assign(&mut self, rhs: Self) {
        let values = Arc::make_mut(&mut self.values);
        for i in 0..DENSE_SPECTRUM_SAMPLES {
            values[i] *= rhs.values[i];
        }
    }
}

impl std::ops::Mul for DenselySampledSpectrum {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl std::ops::Mul<Float> for DenselySampledSpectrum {
    type Output = Self;

    fn mul(mut self, rhs: Float) -> Self::Output {
        for value in Arc::make_mut(&mut self.values).iter_mut() {
            *value *= rhs;
        }
        self
    }
}

impl std::ops::Mul<DenselySampledSpectrum> for Float {
    type Output = DenselySampledSpectrum;

    fn mul(self, rhs: DenselySampledSpectrum) -> Self::Output {
        rhs * self
    }
}

impl std::ops::DivAssign<Float> for DenselySampledSpectrum {
    fn div_assign(&mut self, rhs: Float) {
        for value in Arc::make_mut(&mut self.values).iter_mut() {
            *value /= rhs;
        }
    }
}

impl std::ops::Div<Float> for DenselySampledSpectrum {
    type Output = Self;

    fn div(mut self, rhs: Float) -> Self::Output {
        self /= rhs;
        self
    }
}

impl std::ops::Neg for DenselySampledSpectrum {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut values = *self.values;
        for value in &mut values {
            *value = -*value;
        }
        Self {
            values: Arc::new(values),
        }
    }
}

impl std::ops::DivAssign for DenselySampledSpectrum {
    fn div_assign(&mut self, rhs: Self) {
        let values = Arc::make_mut(&mut self.values);
        for i in 0..DENSE_SPECTRUM_SAMPLES {
            values[i] = if rhs.values[i] == 0.0 {
                0.0
            } else {
                values[i] / rhs.values[i]
            };
        }
    }
}

impl std::ops::Div for DenselySampledSpectrum {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}
