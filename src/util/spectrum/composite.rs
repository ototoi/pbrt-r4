use super::blackbody::BlackbodySpectrum;
use super::constant::ConstantSpectrum;
use super::densely_sampled::DenselySampledSpectrum;
use super::helpers::{sort_spectrum_samples, spectrum_samples_sorted};
use super::piecewise_linear::PiecewiseLinearSpectrum;
use super::rgb_albedo::RGBAlbedoSpectrum;
use super::rgb_illuminant::RGBIlluminantSpectrum;
use super::rgb_to_spectrum::RGBColorSpace;
use super::rgb_unbounded::RGBUnboundedSpectrum;
use super::sampled::{SampledSpectrum, SampledWavelengths};
use super::source::SpectrumType;
use crate::util::base::Float;
use std::ops;

#[derive(Debug, Clone, PartialEq)]
pub enum Spectrum {
    Blackbody(BlackbodySpectrum),
    RGBAlbedo(RGBAlbedoSpectrum),
    RGBUnbounded(RGBUnboundedSpectrum),
    RGBIlluminant(RGBIlluminantSpectrum),
    PiecewiseLinear(PiecewiseLinearSpectrum),
    Constant(ConstantSpectrum),
    DenselySampled(DenselySampledSpectrum),
}

impl Spectrum {
    pub const N_SAMPLES: usize = DenselySampledSpectrum::N_SAMPLES;

    pub fn zero() -> Self {
        Self::Constant(ConstantSpectrum::new(0.0))
    }

    pub fn one() -> Self {
        Self::Constant(ConstantSpectrum::new(1.0))
    }

    pub fn from_rgb_albedo(rgb: &[Float]) -> Self {
        Self::RGBAlbedo(RGBAlbedoSpectrum::new([rgb[0], rgb[1], rgb[2]]))
    }

    pub fn from_rgb(rgb: &[Float], spectrum_type: SpectrumType) -> Self {
        match spectrum_type {
            SpectrumType::Albedo => Self::from_rgb_albedo(rgb),
            SpectrumType::Unbounded => Self::from_rgb_unbounded(rgb),
            SpectrumType::Illuminant => Self::from_rgb_illuminant(rgb),
        }
    }

    pub fn from_rgb_in_color_space(
        color_space: &RGBColorSpace,
        rgb: &[Float],
        spectrum_type: SpectrumType,
    ) -> Self {
        let rgb = [rgb[0], rgb[1], rgb[2]];
        match spectrum_type {
            SpectrumType::Albedo => {
                Self::RGBAlbedo(RGBAlbedoSpectrum::from_color_space(color_space, rgb))
            }
            SpectrumType::Unbounded => {
                Self::RGBUnbounded(RGBUnboundedSpectrum::from_color_space(color_space, rgb))
            }
            SpectrumType::Illuminant => {
                Self::RGBIlluminant(RGBIlluminantSpectrum::from_color_space(color_space, rgb))
            }
        }
    }

    pub fn from_rgb_reflectance(rgb: &[Float]) -> Self {
        Self::from_rgb_albedo(rgb)
    }

    pub fn from_rgb_unbounded(rgb: &[Float]) -> Self {
        Self::RGBUnbounded(RGBUnboundedSpectrum::new([rgb[0], rgb[1], rgb[2]]))
    }

    pub fn from_rgb_illuminant(rgb: &[Float]) -> Self {
        Self::RGBIlluminant(RGBIlluminantSpectrum::new([rgb[0], rgb[1], rgb[2]]))
    }

    pub fn from_sampled(lambda: &[Float], values: &[Float]) -> Self {
        let mut lambda = lambda.to_vec();
        let mut values = values.to_vec();

        if !spectrum_samples_sorted(&lambda, &values) {
            sort_spectrum_samples(&mut lambda, &mut values);
        }

        let mut unique_lambda: Vec<Float> = Vec::with_capacity(lambda.len());
        let mut unique_values: Vec<Float> = Vec::with_capacity(values.len());
        for (l, v) in lambda.into_iter().zip(values.into_iter()) {
            if let Some(last) = unique_lambda.last() {
                if (*last - l).abs() <= Float::EPSILON {
                    if let Some(last_value) = unique_values.last_mut() {
                        *last_value = v;
                    }
                    continue;
                }
            }
            unique_lambda.push(l);
            unique_values.push(v);
        }

        Self::PiecewiseLinear(PiecewiseLinearSpectrum::new(unique_lambda, unique_values))
    }

    pub fn len(&self) -> usize {
        Self::N_SAMPLES
    }

    pub fn is_constant_spectrum(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    pub fn is_valid(&self) -> bool {
        match self {
            Self::Blackbody(s) => s.max_value().is_finite(),
            Self::RGBAlbedo(s) => s.max_value().is_finite(),
            Self::RGBUnbounded(s) => s.max_value().is_finite(),
            Self::RGBIlluminant(s) => s.max_value().is_finite(),
            Self::PiecewiseLinear(s) => s.max_value().is_finite(),
            Self::Constant(s) => s.value.is_finite(),
            Self::DenselySampled(s) => s.is_valid(),
        }
    }

    pub fn sample_at(&self, lambda: Float) -> Float {
        match self {
            Self::Blackbody(s) => s.sample_at(lambda),
            Self::RGBAlbedo(s) => s.sample_at(lambda),
            Self::RGBUnbounded(s) => s.sample_at(lambda),
            Self::RGBIlluminant(s) => s.sample_at(lambda),
            Self::PiecewiseLinear(s) => s.sample_at(lambda),
            Self::Constant(s) => s.value,
            Self::DenselySampled(s) => s.sample_at(lambda),
        }
    }

    pub fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Self::Blackbody(s) => s.sample(lambda),
            Self::RGBAlbedo(s) => s.sample(lambda),
            Self::RGBUnbounded(s) => s.sample(lambda),
            Self::RGBIlluminant(s) => s.sample(lambda),
            Self::PiecewiseLinear(s) => s.sample(lambda),
            Self::Constant(s) => s.sample(lambda),
            Self::DenselySampled(s) => s.sample(lambda),
        }
    }

    /// Build an RGB-derived `SampledSpectrum` directly without
    /// materializing the intermediate `Spectrum` enum. Mirrors v4's
    /// `RGBAlbedoSpectrum(*colorSpace, rgb).Sample(lambda)` pattern in
    /// `ImageTextureBase::Evaluate` — used on the texture-shade hot path
    /// to avoid one match-dispatch + Spectrum allocation per shade.
    pub fn rgb_to_sampled(
        rgb: [Float; 3],
        spectrum_type: SpectrumType,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        match spectrum_type {
            SpectrumType::Albedo => super::rgb_albedo::RGBAlbedoSpectrum::new(rgb).sample(lambda),
            SpectrumType::Unbounded => {
                super::rgb_unbounded::RGBUnboundedSpectrum::new(rgb).sample(lambda)
            }
            SpectrumType::Illuminant => {
                super::rgb_illuminant::RGBIlluminantSpectrum::new(rgb).sample(lambda)
            }
        }
    }

    pub fn to_dense(&self) -> DenselySampledSpectrum {
        DenselySampledSpectrum::from_spectrum(self)
    }

    pub fn clamp(&self, low: Float, high: Float) -> Self {
        Self::DenselySampled(self.to_dense().clamp(low, high))
    }

    pub fn clamp_zero(&self) -> Self {
        Self::DenselySampled(self.to_dense().clamp_zero())
    }

    pub fn sqrt(&self) -> Self {
        Self::DenselySampled(self.to_dense().sqrt())
    }

    pub fn exp(&self) -> Self {
        Self::DenselySampled(self.to_dense().exp())
    }

    pub fn average(&self) -> Float {
        self.to_dense().average()
    }

    pub fn y(&self) -> Float {
        self.to_dense().y()
    }

    pub fn max_component_value(&self) -> Float {
        match self {
            Self::Blackbody(s) => s.max_value(),
            Self::RGBAlbedo(s) => s.max_value(),
            Self::RGBUnbounded(s) => s.max_value(),
            Self::RGBIlluminant(s) => s.max_value(),
            Self::PiecewiseLinear(s) => s.max_value(),
            Self::Constant(s) => s.max_value(),
            Self::DenselySampled(s) => s.max_value(),
        }
    }

    pub fn max_value(&self) -> Float {
        self.max_component_value()
    }

    pub fn to_rgb(&self) -> [Float; 3] {
        self.to_dense().to_rgb()
    }

    pub fn is_black(&self) -> bool {
        self.max_component_value() <= 0.0
    }

    pub fn at(&self, index: usize) -> Float {
        self.to_dense()[index]
    }
}

pub fn inner_product(f: &Spectrum, g: &Spectrum) -> Float {
    let mut integral = 0.0;
    for lambda in 0..Spectrum::N_SAMPLES {
        integral += f.at(lambda) * g.at(lambda);
    }
    integral
}

pub fn spectrum_to_photometric(s: &Spectrum) -> Float {
    // pbrt-v4 `SpectrumToPhotometric` (spectrum.cpp:37-47): for
    // RGBIlluminantSpectrum the integral is over the underlying
    // illuminant only, with the RGB scale and reflectance polynomial
    // stripped off, so the photometric divisor stays independent of
    // the RGB magnitude (otherwise lights get scaled twice).
    if let Spectrum::RGBIlluminant(rgb) = s {
        let illum_dense = rgb.illuminant_dense();
        return illum_dense.y() * super::cie::CIE_Y_INTEGRAL;
    }
    s.y() * super::cie::CIE_Y_INTEGRAL
}

pub fn spectrum_to_xyz(s: &Spectrum) -> [Float; 3] {
    s.to_dense().to_xyz()
}

impl From<Float> for Spectrum {
    fn from(value: Float) -> Self {
        Self::Constant(ConstantSpectrum::new(value))
    }
}

impl From<[Float; 3]> for Spectrum {
    fn from(rgb: [Float; 3]) -> Self {
        Self::from_rgb_unbounded(&rgb)
    }
}

impl From<&[Float; 3]> for Spectrum {
    fn from(rgb: &[Float; 3]) -> Self {
        Self::from_rgb_unbounded(rgb)
    }
}

impl From<DenselySampledSpectrum> for Spectrum {
    fn from(value: DenselySampledSpectrum) -> Self {
        Self::DenselySampled(value)
    }
}

impl From<&DenselySampledSpectrum> for Spectrum {
    fn from(value: &DenselySampledSpectrum) -> Self {
        Self::DenselySampled(value.clone())
    }
}

impl Default for Spectrum {
    fn default() -> Self {
        Self::zero()
    }
}

impl ops::Add for Spectrum {
    type Output = Spectrum;

    fn add(self, rhs: Self) -> Self::Output {
        Spectrum::DenselySampled(self.to_dense() + rhs.to_dense())
    }
}

impl ops::Sub for Spectrum {
    type Output = Spectrum;

    fn sub(self, rhs: Self) -> Self::Output {
        Spectrum::DenselySampled(self.to_dense() - rhs.to_dense())
    }
}

impl ops::Mul for Spectrum {
    type Output = Spectrum;

    fn mul(self, rhs: Self) -> Self::Output {
        Spectrum::DenselySampled(self.to_dense() * rhs.to_dense())
    }
}

impl ops::Mul<Float> for Spectrum {
    type Output = Spectrum;

    fn mul(self, rhs: Float) -> Self::Output {
        Spectrum::DenselySampled(self.to_dense() * rhs)
    }
}

impl ops::Mul<Spectrum> for Float {
    type Output = Spectrum;

    fn mul(self, rhs: Spectrum) -> Self::Output {
        rhs * self
    }
}

impl ops::Div<Float> for Spectrum {
    type Output = Spectrum;

    fn div(self, rhs: Float) -> Self::Output {
        Spectrum::DenselySampled(self.to_dense() / rhs)
    }
}

impl ops::Div for Spectrum {
    type Output = Spectrum;

    fn div(self, rhs: Self) -> Self::Output {
        Spectrum::DenselySampled(self.to_dense() / rhs.to_dense())
    }
}

impl ops::Neg for Spectrum {
    type Output = Spectrum;

    fn neg(self) -> Self::Output {
        Spectrum::DenselySampled(self.to_dense() * -1.0)
    }
}

impl ops::AddAssign for Spectrum {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl ops::MulAssign for Spectrum {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl ops::MulAssign<Float> for Spectrum {
    fn mul_assign(&mut self, rhs: Float) {
        *self = self.clone() * rhs;
    }
}

impl ops::DivAssign<Float> for Spectrum {
    fn div_assign(&mut self, rhs: Float) {
        *self = self.clone() / rhs;
    }
}

impl ops::DivAssign for Spectrum {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}
