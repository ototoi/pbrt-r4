use super::config::{LAMBDA_MAX, LAMBDA_MIN, N_SPECTRUM_SAMPLES};
use super::sampled::{SampledSpectrum, SampledWavelengths};

include!(concat!(env!("OUT_DIR"), "/spectrum_data_cie.rs"));

pub const VISIBLE_LAMBDA_MIN: Float = LAMBDA_MIN as Float;
pub const VISIBLE_LAMBDA_MAX: Float = LAMBDA_MAX as Float;

#[inline]
pub fn lerp(t: Float, a: Float, b: Float) -> Float {
    a + (b - a) * t
}

pub fn sample_dense_array(values: &[Float], lambda: Float) -> Float {
    if lambda <= CIE_LAMBDA[0] {
        return values[0];
    }
    if lambda >= CIE_LAMBDA[CIE_SAMPLES - 1] {
        return values[CIE_SAMPLES - 1];
    }

    let offset = lambda - CIE_LAMBDA[0];
    let index = offset.floor() as usize;
    let t = offset - index as Float;
    let idx0 = index.min(CIE_SAMPLES - 1);
    let idx1 = (idx0 + 1).min(CIE_SAMPLES - 1);
    lerp(t, values[idx0], values[idx1])
}

pub fn sample_cie_x(lambda: &SampledWavelengths) -> SampledSpectrum {
    let mut values = [0.0; N_SPECTRUM_SAMPLES];
    for i in 0..N_SPECTRUM_SAMPLES {
        values[i] = sample_dense_array(&CIE_X, lambda[i]);
    }
    SampledSpectrum::from(values)
}

pub fn sample_cie_y(lambda: &SampledWavelengths) -> SampledSpectrum {
    let mut values = [0.0; N_SPECTRUM_SAMPLES];
    for i in 0..N_SPECTRUM_SAMPLES {
        values[i] = sample_dense_array(&CIE_Y, lambda[i]);
    }
    SampledSpectrum::from(values)
}

pub fn sample_cie_z(lambda: &SampledWavelengths) -> SampledSpectrum {
    let mut values = [0.0; N_SPECTRUM_SAMPLES];
    for i in 0..N_SPECTRUM_SAMPLES {
        values[i] = sample_dense_array(&CIE_Z, lambda[i]);
    }
    SampledSpectrum::from(values)
}

pub fn xyz_to_rgb(xyz: &[Float; 3]) -> [Float; 3] {
    [
        3.240479 * xyz[0] - 1.537150 * xyz[1] - 0.498535 * xyz[2],
        -0.969256 * xyz[0] + 1.875991 * xyz[1] + 0.041556 * xyz[2],
        0.055648 * xyz[0] - 0.204043 * xyz[1] + 1.057311 * xyz[2],
    ]
}

pub fn visible_wavelengths_pdf(lambda: Float) -> Float {
    if !(VISIBLE_LAMBDA_MIN..=VISIBLE_LAMBDA_MAX).contains(&lambda) {
        return 0.0;
    }
    0.003_939_804_2 / (0.0072 * (lambda - 538.0)).cosh().powi(2)
}

pub fn sample_visible_wavelengths(u: Float) -> Float {
    538.0 - 138.888_889 * (0.856_910_62 - 1.827_501_97 * u).atanh()
}
