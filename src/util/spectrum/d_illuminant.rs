//! CIE D-illuminant generator (pbrt-v4 `Spectra::D`).
//!
//! Used by `PixelSensor::create` to build the spectral power
//! distribution of the source white point from the scene's
//! `whitebalance` parameter so the sensor's `xyz_from_sensor_rgb`
//! matrix can be re-derived at runtime (the precomputed build-time
//! matrix only handles whitebalance = 6500 K).

use super::blackbody::BlackbodySpectrum;
use super::piecewise_linear::PiecewiseLinearSpectrum;
use super::sampled::SampledWavelengths;
use crate::util::base::*;

pub const N_CIE_S: usize = 107;

#[rustfmt::skip]
pub const CIE_S_LAMBDA: [Float; N_CIE_S] = [
    300.0, 305.0, 310.0, 315.0, 320.0, 325.0, 330.0, 335.0, 340.0, 345.0,
    350.0, 355.0, 360.0, 365.0, 370.0, 375.0, 380.0, 385.0, 390.0, 395.0,
    400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 440.0, 445.0,
    450.0, 455.0, 460.0, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0,
    500.0, 505.0, 510.0, 515.0, 520.0, 525.0, 530.0, 535.0, 540.0, 545.0,
    550.0, 555.0, 560.0, 565.0, 570.0, 575.0, 580.0, 585.0, 590.0, 595.0,
    600.0, 605.0, 610.0, 615.0, 620.0, 625.0, 630.0, 635.0, 640.0, 645.0,
    650.0, 655.0, 660.0, 665.0, 670.0, 675.0, 680.0, 685.0, 690.0, 695.0,
    700.0, 705.0, 710.0, 715.0, 720.0, 725.0, 730.0, 735.0, 740.0, 745.0,
    750.0, 755.0, 760.0, 765.0, 770.0, 775.0, 780.0, 785.0, 790.0, 795.0,
    800.0, 805.0, 810.0, 815.0, 820.0, 825.0, 830.0,
];

#[rustfmt::skip]
pub const CIE_S0: [Float; N_CIE_S] = [
    0.04, 3.02, 6.0, 17.8, 29.6, 42.45, 55.3, 56.3, 57.3, 59.55,
    61.8, 61.65, 61.5, 65.15, 68.8, 66.1, 63.4, 64.6, 65.8, 80.3,
    94.8, 99.8, 104.8, 105.35, 105.9, 101.35, 96.8, 105.35, 113.9, 119.75,
    125.6, 125.55, 125.5, 123.4, 121.3, 121.3, 121.3, 117.4, 113.5, 113.3,
    113.1, 111.95, 110.8, 108.65, 106.5, 107.65, 108.8, 107.05, 105.3, 104.85,
    104.4, 102.2, 100.0, 98.0, 96.0, 95.55, 95.1, 92.1, 89.1, 89.8,
    90.5, 90.4, 90.3, 89.35, 88.4, 86.2, 84.0, 84.55, 85.1, 83.5,
    81.9, 82.25, 82.6, 83.75, 84.9, 83.1, 81.3, 76.6, 71.9, 73.1,
    74.3, 75.35, 76.4, 69.85, 63.3, 67.5, 71.7, 74.35, 77.0, 71.1,
    65.2, 56.45, 47.7, 58.15, 68.6, 66.8, 65.0, 65.5, 66.0, 63.5,
    61.0, 57.15, 53.3, 56.1, 58.9, 60.4, 61.9,
];

#[rustfmt::skip]
pub const CIE_S1: [Float; N_CIE_S] = [
    0.02, 2.26, 4.5, 13.45, 22.4, 32.2, 42.0, 41.3, 40.6, 41.1,
    41.6, 39.8, 38.0, 40.2, 42.4, 40.45, 38.5, 36.75, 35.0, 39.2,
    43.4, 44.85, 46.3, 45.1, 43.9, 40.5, 37.1, 36.9, 36.7, 36.3,
    35.9, 34.25, 32.6, 30.25, 27.9, 26.1, 24.3, 22.2, 20.1, 18.15,
    16.2, 14.7, 13.2, 10.9, 8.6, 7.35, 6.1, 5.15, 4.2, 3.05,
    1.9, 0.95, 0.0, -0.8, -1.6, -2.55, -3.5, -3.5, -3.5, -4.65,
    -5.8, -6.5, -7.2, -7.9, -8.6, -9.05, -9.5, -10.2, -10.9, -10.8,
    -10.7, -11.35, -12.0, -13.0, -14.0, -13.8, -13.6, -12.8, -12.0, -12.65,
    -13.3, -13.1, -12.9, -11.75, -10.6, -11.1, -11.6, -11.9, -12.2, -11.2,
    -10.2, -9.0, -7.8, -9.5, -11.2, -10.8, -10.4, -10.5, -10.6, -10.15,
    -9.7, -9.0, -8.3, -8.8, -9.3, -9.55, -9.8,
];

#[rustfmt::skip]
pub const CIE_S2: [Float; N_CIE_S] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.25, 8.5, 8.15, 7.8, 7.25,
    6.7, 6.0, 5.3, 5.7, 6.1, 4.55, 3.0, 2.1, 1.2, 0.05,
    -1.1, -0.8, -0.5, -0.6, -0.7, -0.95, -1.2, -1.9, -2.6, -2.75,
    -2.9, -2.85, -2.8, -2.7, -2.6, -2.6, -2.6, -2.2, -1.8, -1.65,
    -1.5, -1.4, -1.3, -1.25, -1.2, -1.1, -1.0, -0.75, -0.5, -0.4,
    -0.3, -0.15, 0.0, 0.1, 0.2, 0.35, 0.5, 1.3, 2.1, 2.65,
    3.2, 3.65, 4.1, 4.4, 4.7, 4.9, 5.1, 5.9, 6.7, 7.0,
    7.3, 7.95, 8.6, 9.2, 9.8, 10.0, 10.2, 9.25, 8.3, 8.95,
    9.6, 9.05, 8.5, 7.75, 7.0, 7.3, 7.6, 7.8, 8.0, 7.35,
    6.7, 5.95, 5.2, 6.3, 7.4, 7.1, 6.8, 6.9, 7.0, 6.7,
    6.4, 5.95, 5.5, 5.8, 6.1, 6.3, 6.5,
];

/// pbrt-v4 `Spectra::D` (`util/spectrum.cpp:2533`). Returns the CIE
/// D-series daylight illuminant for the given correlated color
/// temperature in Kelvin, as a `PiecewiseLinearSpectrum` so callers
/// can sample it at arbitrary wavelengths. Below 4000 K the D series
/// is ill-defined, so falls back to a `BlackbodySpectrum`-driven
/// piecewise-linear approximation sampled at `CIE_S_LAMBDA`.
pub fn d_illuminant(temperature: Float) -> PiecewiseLinearSpectrum {
    // CCT correction baked into the v4 implementation: matches the
    // `1.4388 / 1.4380` factor in pbrt-v4 (the ratio of the c2 second
    // radiation constant from the CIE 1968 table to the value used in
    // the older D-illuminant tables).
    let cct = temperature * 1.4388 / 1.4380;

    if cct < 4000.0 {
        let bb = BlackbodySpectrum::new(cct, 1.0);
        let mut values = Vec::with_capacity(N_CIE_S);
        for i in 0..N_CIE_S {
            values.push(bb.sample_at(CIE_S_LAMBDA[i]));
        }
        return PiecewiseLinearSpectrum::new(CIE_S_LAMBDA.to_vec(), values);
    }

    // CCT -> xy chromaticity via the standard cubic fits.
    let x = if cct <= 7000.0 {
        -4.607 * 1e9 / cct.powi(3) + 2.9678 * 1e6 / (cct * cct) + 0.09911 * 1e3 / cct + 0.244063
    } else {
        -2.0064 * 1e9 / cct.powi(3) + 1.9018 * 1e6 / (cct * cct) + 0.24748 * 1e3 / cct + 0.23704
    };
    let y = -3.0 * x * x + 2.870 * x - 0.275;

    let m = 0.0241 + 0.2562 * x - 0.7341 * y;
    let m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / m;
    let m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / m;

    let mut values = Vec::with_capacity(N_CIE_S);
    for i in 0..N_CIE_S {
        values.push((CIE_S0[i] + CIE_S1[i] * m1 + CIE_S2[i] * m2) * 0.01);
    }
    PiecewiseLinearSpectrum::new(CIE_S_LAMBDA.to_vec(), values)
}

/// Convenience: sample a D-illuminant directly into a
/// `SampledWavelengths` packet.
pub fn sample_d_illuminant(
    temperature: Float,
    lambda: &SampledWavelengths,
) -> super::sampled::SampledSpectrum {
    d_illuminant(temperature).sample(lambda)
}
