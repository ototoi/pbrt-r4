// Math helpers used by the sensor build step:
// piecewise-linear -> 1nm dense sampling (mirrors pbrt-v4
// `PiecewiseLinearSpectrum -> DenselySampledSpectrum`),
// the `ProjectReflectance<Triplet>` template (film.h:120), and a
// 3x3 invert + 24x3 -> 3x3 linear least squares matching v4's
// `LinearLeastSquares` (util/math.cpp).

use crate::spectrum::config::*;

pub const LAMBDA_MIN_F: Float = LAMBDA_MIN as Float;
pub const LAMBDA_MAX_F: Float = LAMBDA_MAX as Float;
pub const DENSE_N: usize = (LAMBDA_MAX - LAMBDA_MIN + 1) as usize;

/// Sample a piecewise-linear (lambda, value) curve at `lambda`.
/// Returns 0 outside [lambda_arr.first(), lambda_arr.last()] to
/// match pbrt-v4 `PiecewiseLinearSpectrum::operator()(lambda)`.
pub fn pwl_eval(lambda_arr: &[Float], values: &[Float], lambda: Float) -> Float {
    assert_eq!(lambda_arr.len(), values.len());
    let n = lambda_arr.len();
    if n == 0 || lambda < lambda_arr[0] || lambda > lambda_arr[n - 1] {
        return 0.0;
    }
    // Binary search for the bracket.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if lambda_arr[mid] <= lambda {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (lambda - lambda_arr[lo]) / (lambda_arr[hi] - lambda_arr[lo]);
    values[lo] + t * (values[hi] - values[lo])
}

/// Densify a piecewise-linear spectrum to 1nm samples over
/// [LAMBDA_MIN, LAMBDA_MAX]. Same convention as r4's runtime
/// `DenselySampledSpectrum::from_sampled`.
pub fn densify(lambda_arr: &[Float], values: &[Float]) -> Vec<Float> {
    (0..DENSE_N)
        .map(|i| pwl_eval(lambda_arr, values, LAMBDA_MIN_F + i as Float))
        .collect()
}

/// 1nm-step inner product over [LAMBDA_MIN, LAMBDA_MAX]. Both
/// inputs are dense samples produced by [`densify`]. The step is 1
/// so the sum doubles as the Riemann integral.
pub fn inner_product(a: &[Float], b: &[Float]) -> Float {
    assert_eq!(a.len(), DENSE_N);
    assert_eq!(b.len(), DENSE_N);
    (0..DENSE_N).map(|i| a[i] * b[i]).sum()
}

/// pbrt-v4 `PixelSensor::ProjectReflectance<Triplet>(refl, illum,
/// b1, b2, b3)` (film.h:120). All inputs are dense 1nm samples.
/// Returns the triplet normalized by `g_integral = ∫ b2 · illum`
/// so that a white reflector (refl ≡ 1) gives Triplet[1] == 1.
pub fn project_reflectance(
    refl: &[Float],
    illum: &[Float],
    b1: &[Float],
    b2: &[Float],
    b3: &[Float],
) -> [Float; 3] {
    let mut g_integral = 0.0;
    let mut r = [0.0; 3];
    for i in 0..DENSE_N {
        let il = illum[i];
        g_integral += b2[i] * il;
        let ri = refl[i] * il;
        r[0] += b1[i] * ri;
        r[1] += b2[i] * ri;
        r[2] += b3[i] * ri;
    }
    [r[0] / g_integral, r[1] / g_integral, r[2] / g_integral]
}

/// 3x3 matrix inverse via cofactor expansion. Returns `None` if
/// the determinant is below `1e-12`.
pub fn invert3(m: &[[Float; 3]; 3]) -> Option<[[Float; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut out = [[0.0; 3]; 3];
    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    Some(out)
}

/// pbrt-v4 `LinearLeastSquares(A, B, rows)` for 24x3 -> 3x3.
/// Solves min |A·M^T - B|^2 (per-row dot products with M's rows)
/// by the normal equations: M^T = (A^T A)^{-1} A^T B.
/// `a[i][3]` is row i of A (camera RGB per swatch), `b[i][3]` is
/// row i of B (target XYZ per swatch). Returns the 3x3 matrix M
/// such that `M · a[i] ≈ b[i]`.
pub fn linear_least_squares_3x3(a: &[[Float; 3]], b: &[[Float; 3]]) -> Option<[[Float; 3]; 3]> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    // ata = A^T · A (3x3)
    let mut ata = [[0.0; 3]; 3];
    for row in a.iter().take(n) {
        for j in 0..3 {
            for k in 0..3 {
                ata[j][k] += row[j] * row[k];
            }
        }
    }
    let ata_inv = invert3(&ata)?;
    // atb = A^T · B (3x3)
    let mut atb = [[0.0; 3]; 3];
    for i in 0..n {
        for j in 0..3 {
            for k in 0..3 {
                atb[j][k] += a[i][j] * b[i][k];
            }
        }
    }
    // M^T = ata_inv · atb -> M[j][k] = ata_inv[k][i] · atb[i][j]
    let mut m = [[0.0; 3]; 3];
    for j in 0..3 {
        for k in 0..3 {
            let mut s = 0.0;
            for i in 0..3 {
                s += ata_inv[k][i] * atb[i][j];
            }
            m[j][k] = s;
        }
    }
    Some(m)
}
