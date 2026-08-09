use super::cie::{
    lerp, sample_cie_x, sample_cie_y, sample_cie_z, sample_visible_wavelengths,
    visible_wavelengths_pdf, xyz_to_rgb, CIE_Y_INTEGRAL,
};
use super::composite::Spectrum;
use super::config::{LAMBDA_MAX, LAMBDA_MIN, N_SPECTRUM_SAMPLES};
use crate::util::base::Float;

pub const R4_SPECTRUM_LAMBDA_MIN: Float = LAMBDA_MIN as Float;
pub const R4_SPECTRUM_LAMBDA_MAX: Float = LAMBDA_MAX as Float;

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct SampledSpectrum {
    values: [Float; N_SPECTRUM_SAMPLES],
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SampledWavelengths {
    lambda: [Float; N_SPECTRUM_SAMPLES],
    pdf: [Float; N_SPECTRUM_SAMPLES],
}

impl Default for SampledWavelengths {
    fn default() -> Self {
        Self::sample_uniform(0.5, R4_SPECTRUM_LAMBDA_MIN, R4_SPECTRUM_LAMBDA_MAX)
    }
}

impl SampledSpectrum {
    pub const N_SAMPLES: usize = N_SPECTRUM_SAMPLES;

    pub fn zero() -> Self {
        Self::new(0.0)
    }

    pub fn one() -> Self {
        Self::new(1.0)
    }

    pub fn new(c: Float) -> Self {
        Self {
            values: [c; N_SPECTRUM_SAMPLES],
        }
    }

    pub fn from_slice(v: &[Float]) -> Self {
        assert_eq!(v.len(), N_SPECTRUM_SAMPLES);
        let mut values = [0.0; N_SPECTRUM_SAMPLES];
        values.copy_from_slice(v);
        Self { values }
    }

    pub fn clamp(&self, low: Float, high: Float) -> Self {
        let mut values = self.values;
        for value in &mut values {
            *value = value.clamp(low, high);
        }
        Self { values }
    }

    pub fn clamp_zero(&self) -> Self {
        self.clamp(0.0, Float::INFINITY)
    }

    pub fn has_nans(&self) -> bool {
        self.values.iter().any(|v| v.is_nan())
    }

    pub fn is_black(&self) -> bool {
        !self.values.iter().any(|v| *v != 0.0)
    }

    pub fn is_valid(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }

    pub fn min_component_value(&self) -> Float {
        let mut m = self.values[0];
        for i in 1..N_SPECTRUM_SAMPLES {
            m = Float::min(m, self.values[i]);
        }
        m
    }

    pub fn max_component_value(&self) -> Float {
        let mut m = self.values[0];
        for i in 1..N_SPECTRUM_SAMPLES {
            m = Float::max(m, self.values[i]);
        }
        m
    }

    pub fn average(&self) -> Float {
        let mut sum = self.values[0];
        for i in 1..N_SPECTRUM_SAMPLES {
            sum += self.values[i];
        }
        sum / N_SPECTRUM_SAMPLES as Float
    }

    pub fn sqrt(&self) -> Self {
        let mut values = self.values;
        for value in &mut values {
            *value = value.sqrt();
        }
        Self { values }
    }

    pub fn exp(&self) -> Self {
        let mut values = self.values;
        for value in &mut values {
            *value = value.exp();
        }
        Self { values }
    }

    pub fn from_pdf(lambda: &SampledWavelengths) -> Self {
        Self { values: lambda.pdf }
    }

    pub fn sample(&self, _lambda: &SampledWavelengths) -> Self {
        *self
    }

    pub fn to_dense(&self, lambda: &SampledWavelengths) -> Spectrum {
        Spectrum::from_sampled(lambda.lambda(), &self.values)
    }

    pub fn y(&self, lambda: &SampledWavelengths) -> Float {
        let ys = sample_cie_y(lambda);
        safe_div(ys * *self, lambda.pdf_spectrum()).average() / CIE_Y_INTEGRAL
    }

    pub fn to_xyz(&self, lambda: &SampledWavelengths) -> [Float; 3] {
        let x = sample_cie_x(lambda);
        let y = sample_cie_y(lambda);
        let z = sample_cie_z(lambda);
        let pdf = lambda.pdf_spectrum();

        [
            safe_div(x * *self, pdf).average() / CIE_Y_INTEGRAL,
            safe_div(y * *self, pdf).average() / CIE_Y_INTEGRAL,
            safe_div(z * *self, pdf).average() / CIE_Y_INTEGRAL,
        ]
    }

    pub fn to_rgb(&self, lambda: &SampledWavelengths) -> [Float; 3] {
        xyz_to_rgb(&self.to_xyz(lambda))
    }

    pub fn near_equal(a: &Self, b: &Self, eps: Float) -> bool {
        for i in 0..N_SPECTRUM_SAMPLES {
            if (a.values[i] - b.values[i]).abs() > eps {
                return false;
            }
        }
        true
    }

    pub fn values(&self) -> &[Float; N_SPECTRUM_SAMPLES] {
        &self.values
    }
}

impl SampledWavelengths {
    pub fn sample_uniform(u: Float, lambda_min: Float, lambda_max: Float) -> Self {
        let mut swl = Self {
            lambda: [0.0; N_SPECTRUM_SAMPLES],
            pdf: [0.0; N_SPECTRUM_SAMPLES],
        };

        swl.lambda[0] = lerp(u, lambda_min, lambda_max);

        let delta = (lambda_max - lambda_min) / N_SPECTRUM_SAMPLES as Float;
        for i in 1..N_SPECTRUM_SAMPLES {
            swl.lambda[i] = swl.lambda[i - 1] + delta;
            if swl.lambda[i] > lambda_max {
                swl.lambda[i] = lambda_min + (swl.lambda[i] - lambda_max);
            }
        }

        for i in 0..N_SPECTRUM_SAMPLES {
            swl.pdf[i] = 1.0 / (lambda_max - lambda_min);
        }

        swl
    }

    pub fn sample_visible(u: Float) -> Self {
        let mut swl = Self {
            lambda: [0.0; N_SPECTRUM_SAMPLES],
            pdf: [0.0; N_SPECTRUM_SAMPLES],
        };
        for i in 0..N_SPECTRUM_SAMPLES {
            let mut up = u + i as Float / N_SPECTRUM_SAMPLES as Float;
            if up > 1.0 {
                up -= 1.0;
            }
            swl.lambda[i] = sample_visible_wavelengths(up);
            swl.pdf[i] = visible_wavelengths_pdf(swl.lambda[i]);
        }
        swl
    }

    pub fn terminate_secondary(&mut self) {
        if self.secondary_terminated() {
            return;
        }
        for i in 1..N_SPECTRUM_SAMPLES {
            self.pdf[i] = 0.0;
        }
        self.pdf[0] /= N_SPECTRUM_SAMPLES as Float;
    }

    pub fn secondary_terminated(&self) -> bool {
        for i in 1..N_SPECTRUM_SAMPLES {
            if self.pdf[i] != 0.0 {
                return false;
            }
        }
        true
    }

    pub fn pdf_spectrum(&self) -> SampledSpectrum {
        SampledSpectrum { values: self.pdf }
    }

    pub fn lambda(&self) -> &[Float; N_SPECTRUM_SAMPLES] {
        &self.lambda
    }

    pub fn pdf(&self) -> &[Float; N_SPECTRUM_SAMPLES] {
        &self.pdf
    }
}

impl std::ops::Index<usize> for SampledSpectrum {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl std::ops::IndexMut<usize> for SampledSpectrum {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl std::ops::Index<usize> for SampledWavelengths {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        &self.lambda[index]
    }
}

impl std::ops::IndexMut<usize> for SampledWavelengths {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.lambda[index]
    }
}

impl From<Float> for SampledSpectrum {
    fn from(value: Float) -> Self {
        Self::new(value)
    }
}

impl From<[Float; N_SPECTRUM_SAMPLES]> for SampledSpectrum {
    fn from(values: [Float; N_SPECTRUM_SAMPLES]) -> Self {
        Self { values }
    }
}

impl std::ops::AddAssign for SampledSpectrum {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N_SPECTRUM_SAMPLES {
            self.values[i] += rhs.values[i];
        }
    }
}

impl std::ops::Add for SampledSpectrum {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl std::ops::SubAssign for SampledSpectrum {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N_SPECTRUM_SAMPLES {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl std::ops::Sub for SampledSpectrum {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl std::ops::Sub<SampledSpectrum> for Float {
    type Output = SampledSpectrum;

    fn sub(self, rhs: SampledSpectrum) -> Self::Output {
        let mut values = [0.0; N_SPECTRUM_SAMPLES];
        for i in 0..N_SPECTRUM_SAMPLES {
            values[i] = self - rhs[i];
        }
        SampledSpectrum::from(values)
    }
}

impl std::ops::Neg for SampledSpectrum {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut values = self.values;
        for value in &mut values {
            *value = -*value;
        }
        Self { values }
    }
}

impl std::ops::MulAssign for SampledSpectrum {
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..N_SPECTRUM_SAMPLES {
            self.values[i] *= rhs.values[i];
        }
    }
}

impl std::ops::Mul for SampledSpectrum {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl std::ops::Mul<Float> for SampledSpectrum {
    type Output = Self;

    fn mul(mut self, rhs: Float) -> Self::Output {
        for value in &mut self.values {
            *value *= rhs;
        }
        self
    }
}

impl std::ops::Mul<SampledSpectrum> for Float {
    type Output = SampledSpectrum;

    fn mul(self, rhs: SampledSpectrum) -> Self::Output {
        rhs * self
    }
}

impl std::ops::MulAssign<Float> for SampledSpectrum {
    fn mul_assign(&mut self, rhs: Float) {
        for value in &mut self.values {
            *value *= rhs;
        }
    }
}

impl std::ops::DivAssign for SampledSpectrum {
    fn div_assign(&mut self, rhs: Self) {
        for i in 0..N_SPECTRUM_SAMPLES {
            if rhs.values[i] == 0.0 {
                self.values[i] = 0.0;
            } else {
                self.values[i] /= rhs.values[i];
            }
        }
    }
}

impl std::ops::Div for SampledSpectrum {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}

impl std::ops::Div<Float> for SampledSpectrum {
    type Output = Self;

    fn div(mut self, rhs: Float) -> Self::Output {
        if rhs == 0.0 {
            return Self::zero();
        }
        for value in &mut self.values {
            *value /= rhs;
        }
        self
    }
}

impl std::ops::DivAssign<Float> for SampledSpectrum {
    fn div_assign(&mut self, rhs: Float) {
        if rhs == 0.0 {
            *self = Self::zero();
            return;
        }
        for value in &mut self.values {
            *value /= rhs;
        }
    }
}

pub fn safe_div(a: SampledSpectrum, b: SampledSpectrum) -> SampledSpectrum {
    let mut values = [0.0; N_SPECTRUM_SAMPLES];
    for i in 0..N_SPECTRUM_SAMPLES {
        values[i] = if b[i] == 0.0 { 0.0 } else { a[i] / b[i] };
    }
    SampledSpectrum::from(values)
}

pub fn clamp_sampled(s: SampledSpectrum, low: Float, high: Float) -> SampledSpectrum {
    s.clamp(low, high)
}

pub fn clamp_zero(s: SampledSpectrum) -> SampledSpectrum {
    s.clamp_zero()
}

pub fn sqrt_sampled(s: SampledSpectrum) -> SampledSpectrum {
    s.sqrt()
}

pub fn safe_sqrt(s: SampledSpectrum) -> SampledSpectrum {
    let mut values = [0.0; N_SPECTRUM_SAMPLES];
    for i in 0..N_SPECTRUM_SAMPLES {
        values[i] = s[i].max(0.0).sqrt();
    }
    SampledSpectrum::from(values)
}

pub fn pow_sampled(s: SampledSpectrum, e: Float) -> SampledSpectrum {
    let mut values = [0.0; N_SPECTRUM_SAMPLES];
    for i in 0..N_SPECTRUM_SAMPLES {
        values[i] = s[i].powf(e);
    }
    SampledSpectrum::from(values)
}

pub fn exp_sampled(s: SampledSpectrum) -> SampledSpectrum {
    s.exp()
}

pub fn fast_exp_sampled(s: SampledSpectrum) -> SampledSpectrum {
    s.exp()
}

pub fn lerp_sampled(t: Float, s0: SampledSpectrum, s1: SampledSpectrum) -> SampledSpectrum {
    (1.0 - t) * s0 + t * s1
}

pub fn bilerp_sampled(p: [Float; 2], v: &[SampledSpectrum]) -> SampledSpectrum {
    assert!(v.len() >= 4);
    (1.0 - p[0]) * (1.0 - p[1]) * v[0]
        + p[0] * (1.0 - p[1]) * v[1]
        + (1.0 - p[0]) * p[1] * v[2]
        + p[0] * p[1] * v[3]
}
