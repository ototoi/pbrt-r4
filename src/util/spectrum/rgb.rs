use std::ops;

use crate::util::base::Float;

use super::cie::{xyz_to_rgb, CIE_LAMBDA, CIE_SAMPLES, CIE_X, CIE_Y, CIE_Y_INTEGRAL, CIE_Z};
use super::helpers::{
    interpolate_spectrum_samples, lerp, rgb_to_xyz, sort_spectrum_samples, spectrum_samples_sorted,
};

const Y_WEIGHT: [Float; 3] = [0.212671, 0.715160, 0.072169];

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct RGBSpectrum {
    c: [Float; 3],
}

impl RGBSpectrum {
    pub const N_SAMPLES: usize = 3;

    #[inline]
    pub fn new(r: Float, g: Float, b: Float) -> Self {
        Self { c: [r, g, b] }
    }

    #[inline]
    pub fn zero() -> Self {
        Self { c: [0.0; 3] }
    }

    #[inline]
    pub fn one() -> Self {
        Self { c: [1.0; 3] }
    }

    pub fn clamp(&self, low: Float, high: Float) -> Self {
        let mut c = self.c;
        for value in &mut c {
            *value = value.clamp(low, high);
        }
        Self { c }
    }

    pub fn clamp_zero(&self) -> Self {
        self.clamp(0.0, Float::INFINITY)
    }

    pub fn max_component_value(&self) -> Float {
        self.c[0].max(self.c[1]).max(self.c[2])
    }

    pub fn to_xyz(&self) -> [Float; 3] {
        rgb_to_xyz(&self.c)
    }

    pub fn y(&self) -> Float {
        Y_WEIGHT[0] * self.c[0] + Y_WEIGHT[1] * self.c[1] + Y_WEIGHT[2] * self.c[2]
    }

    pub fn average(&self) -> Float {
        (self.c[0] + self.c[1] + self.c[2]) / 3.0
    }

    pub fn to_rgb(&self) -> [Float; 3] {
        self.c
    }

    #[inline]
    pub fn mul_scalar(&mut self, s: Float) {
        self.c[0] *= s;
        self.c[1] *= s;
        self.c[2] *= s;
    }

    #[inline]
    pub fn div_scalar(&mut self, s: Float) {
        self.c[0] /= s;
        self.c[1] /= s;
        self.c[2] /= s;
    }

    pub fn to_vec(&self) -> Vec<Float> {
        self.c.to_vec()
    }

    pub fn set_vec(&mut self, values: &[Float]) {
        self.c.copy_from_slice(values);
    }

    #[inline]
    pub fn is_rgb(&self) -> bool {
        true
    }

    #[inline]
    pub fn is_black(&self) -> bool {
        self.c.iter().all(|value| *value == 0.0)
    }

    pub fn is_constant_spectrum(&self) -> bool {
        (self.c[0] - self.c[1]).abs() <= 1e-6 && (self.c[1] - self.c[2]).abs() <= 1e-6
    }

    pub fn is_valid(&self) -> bool {
        self.c.iter().all(|value| value.is_finite())
    }

    pub fn rgb_from_xyz(xyz: &[Float]) -> Self {
        Self {
            c: xyz_to_rgb(&[xyz[0], xyz[1], xyz[2]]),
        }
    }

    pub fn rgb_from_sampled(lambda: &[Float], values: &[Float]) -> Self {
        if !spectrum_samples_sorted(lambda, values) {
            let mut sorted_lambda = lambda.to_vec();
            let mut sorted_values = values.to_vec();
            sort_spectrum_samples(&mut sorted_lambda, &mut sorted_values);
            return Self::rgb_from_sampled(&sorted_lambda, &sorted_values);
        }

        let mut xyz = [0.0; 3];
        for i in 0..CIE_SAMPLES {
            let value = interpolate_spectrum_samples(lambda, values, CIE_LAMBDA[i]);
            xyz[0] += value * CIE_X[i];
            xyz[1] += value * CIE_Y[i];
            xyz[2] += value * CIE_Z[i];
        }

        let scale =
            (CIE_LAMBDA[CIE_SAMPLES - 1] - CIE_LAMBDA[0]) / (CIE_Y_INTEGRAL * CIE_SAMPLES as Float);
        xyz[0] *= scale;
        xyz[1] *= scale;
        xyz[2] *= scale;

        Self::rgb_from_xyz(&xyz)
    }

    pub fn rgb_from_rgb(rgb: &[Float]) -> Self {
        Self {
            c: [rgb[0], rgb[1], rgb[2]],
        }
    }

    pub fn from_sampled(lambda: &[Float], values: &[Float]) -> Self {
        Self::rgb_from_sampled(lambda, values)
    }

    pub fn near_equal(a: &Self, b: &Self, eps: Float) -> bool {
        for i in 0..3 {
            if (a.c[i] - b.c[i]).abs() > eps {
                return false;
            }
        }
        true
    }

    pub fn sqrt(&self) -> Self {
        Self {
            c: [self.c[0].sqrt(), self.c[1].sqrt(), self.c[2].sqrt()],
        }
    }

    pub fn exp(&self) -> Self {
        Self {
            c: [self.c[0].exp(), self.c[1].exp(), self.c[2].exp()],
        }
    }

    pub fn len(&self) -> usize {
        3
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn lerp(t: Float, s1: &Self, s2: &Self) -> Self {
        Self {
            c: [
                lerp(t, s1.c[0], s2.c[0]),
                lerp(t, s1.c[1], s2.c[1]),
                lerp(t, s1.c[2], s2.c[2]),
            ],
        }
    }
}

impl ops::Index<usize> for RGBSpectrum {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        &self.c[index]
    }
}

impl ops::IndexMut<usize> for RGBSpectrum {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.c[index]
    }
}

impl ops::Mul<Float> for RGBSpectrum {
    type Output = RGBSpectrum;

    fn mul(self, s: Float) -> RGBSpectrum {
        RGBSpectrum::from([self[0] * s, self[1] * s, self[2] * s])
    }
}

impl ops::Div<Float> for RGBSpectrum {
    type Output = RGBSpectrum;

    fn div(self, s: Float) -> RGBSpectrum {
        RGBSpectrum::from([self[0] / s, self[1] / s, self[2] / s])
    }
}

impl ops::Mul<RGBSpectrum> for Float {
    type Output = RGBSpectrum;

    fn mul(self, rhs: RGBSpectrum) -> RGBSpectrum {
        RGBSpectrum::from([self * rhs[0], self * rhs[1], self * rhs[2]])
    }
}

impl ops::Add<RGBSpectrum> for RGBSpectrum {
    type Output = RGBSpectrum;

    fn add(self, rhs: RGBSpectrum) -> RGBSpectrum {
        RGBSpectrum::from([self[0] + rhs[0], self[1] + rhs[1], self[2] + rhs[2]])
    }
}

impl ops::Sub<RGBSpectrum> for RGBSpectrum {
    type Output = RGBSpectrum;

    fn sub(self, rhs: RGBSpectrum) -> RGBSpectrum {
        RGBSpectrum::from([self[0] - rhs[0], self[1] - rhs[1], self[2] - rhs[2]])
    }
}

impl ops::Mul<RGBSpectrum> for RGBSpectrum {
    type Output = RGBSpectrum;

    fn mul(self, rhs: RGBSpectrum) -> RGBSpectrum {
        RGBSpectrum::from([self[0] * rhs[0], self[1] * rhs[1], self[2] * rhs[2]])
    }
}

impl ops::Div<RGBSpectrum> for RGBSpectrum {
    type Output = RGBSpectrum;

    fn div(self, rhs: RGBSpectrum) -> RGBSpectrum {
        RGBSpectrum::from([self[0] / rhs[0], self[1] / rhs[1], self[2] / rhs[2]])
    }
}

impl ops::AddAssign<RGBSpectrum> for RGBSpectrum {
    fn add_assign(&mut self, rhs: RGBSpectrum) {
        self.c[0] += rhs[0];
        self.c[1] += rhs[1];
        self.c[2] += rhs[2];
    }
}

impl ops::SubAssign<RGBSpectrum> for RGBSpectrum {
    fn sub_assign(&mut self, rhs: RGBSpectrum) {
        self.c[0] -= rhs[0];
        self.c[1] -= rhs[1];
        self.c[2] -= rhs[2];
    }
}

impl ops::MulAssign<Float> for RGBSpectrum {
    fn mul_assign(&mut self, rhs: Float) {
        self.mul_scalar(rhs);
    }
}

impl ops::MulAssign<RGBSpectrum> for RGBSpectrum {
    fn mul_assign(&mut self, rhs: RGBSpectrum) {
        self.c[0] *= rhs[0];
        self.c[1] *= rhs[1];
        self.c[2] *= rhs[2];
    }
}

impl ops::DivAssign<Float> for RGBSpectrum {
    fn div_assign(&mut self, rhs: Float) {
        self.div_scalar(rhs);
    }
}

impl ops::Neg for RGBSpectrum {
    type Output = RGBSpectrum;

    fn neg(self) -> Self::Output {
        Self {
            c: [-self.c[0], -self.c[1], -self.c[2]],
        }
    }
}

impl Default for RGBSpectrum {
    fn default() -> Self {
        RGBSpectrum::zero()
    }
}

impl From<Float> for RGBSpectrum {
    fn from(value: Float) -> Self {
        Self { c: [value; 3] }
    }
}

impl From<(Float, Float, Float)> for RGBSpectrum {
    fn from(value: (Float, Float, Float)) -> Self {
        Self {
            c: [value.0, value.1, value.2],
        }
    }
}

impl From<&[Float; 3]> for RGBSpectrum {
    fn from(value: &[Float; 3]) -> Self {
        Self { c: *value }
    }
}

impl From<[Float; 3]> for RGBSpectrum {
    fn from(value: [Float; 3]) -> Self {
        Self { c: value }
    }
}

impl From<Vec<Float>> for RGBSpectrum {
    fn from(value: Vec<Float>) -> Self {
        Self {
            c: [value[0], value[1], value[2]],
        }
    }
}

impl From<&RGBSpectrum> for RGBSpectrum {
    fn from(value: &RGBSpectrum) -> Self {
        *value
    }
}
