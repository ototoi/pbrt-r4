// Verbatim port of pbrt-v4 `Spectra::D(T, alloc)` (spectrum.cpp:2545-2566):
// build a CIE D-series daylight illuminant from the (S0, S1, S2)
// basis functions for the requested correlated color temperature.

use crate::spectrum::config::Float;

use super::cie_s_data::{CIE_S0, CIE_S1, CIE_S2, CIE_S_LAMBDA};
use super::math;

/// Daylight illuminant D(T), sampled at the CIE_S basis lambdas
/// (5nm spacing, 300-830 nm). Returns the raw 107 values as the
/// PiecewiseLinearSpectrum from which v4 then builds a dense
/// spectrum.
pub fn d_illuminant_pwl(cct: Float) -> [Float; super::cie_s_data::N_CIE_S] {
    let inv_cct = 1.0 / cct;
    let inv_cct2 = inv_cct * inv_cct;
    let inv_cct3 = inv_cct2 * inv_cct;
    let x = if cct <= 7000.0 {
        -4.607e9 * inv_cct3 + 2.9678e6 * inv_cct2 + 0.09911e3 * inv_cct + 0.244063
    } else {
        -2.0064e9 * inv_cct3 + 1.9018e6 * inv_cct2 + 0.24748e3 * inv_cct + 0.23704
    };
    let y = -3.0 * x * x + 2.870 * x - 0.275;
    let denom = 0.0241 + 0.2562 * x - 0.7341 * y;
    let m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / denom;
    let m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / denom;

    let mut out = [0.0; super::cie_s_data::N_CIE_S];
    for i in 0..super::cie_s_data::N_CIE_S {
        out[i] = (CIE_S0[i] + CIE_S1[i] * m1 + CIE_S2[i] * m2) * 0.01;
    }
    out
}

/// Convenience: D(T) densified to 1nm samples over r4's
/// [LAMBDA_MIN, LAMBDA_MAX] range, the form used downstream by
/// `ProjectReflectance`.
pub fn d_illuminant_dense(cct: Float) -> Vec<Float> {
    let raw = d_illuminant_pwl(cct);
    math::densify(&CIE_S_LAMBDA, &raw)
}
