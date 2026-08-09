use crate::util::base::*;
use crate::util::error::*;
use crate::util::spectrum::*;

use super::swatch_reflectances::{swatch_reflectances, N_SWATCH_REFLECTANCES};
use log::*;

pub(super) mod precomputed {
    include!(concat!(env!("OUT_DIR"), "/pixel_sensor_data.rs"));
}

fn lookup_named_sensor(name: &str) -> Option<&'static precomputed::SensorEntry> {
    precomputed::SENSORS.iter().find(|e| e.name == name)
}

pub fn known_sensor_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = precomputed::SENSORS.iter().map(|e| e.name).collect();
    names.push("cie1931");
    names
}

#[derive(Debug, Clone, PartialEq)]
pub struct PixelSensor {
    sensor_name: &'static str,
    imaging_ratio: Float,
    /// pbrt-v4 film.h:112 -- the sensor sensitivity curves. For
    /// `cie1931` these are the CIE 2-degree X/Y/Z matching
    /// functions; for `canon_eos_5d_mkiv` they are the canon
    /// camera spectra emitted by `build/sensor/`.
    r_bar: DenseSampledSpectrum,
    g_bar: DenseSampledSpectrum,
    b_bar: DenseSampledSpectrum,
    /// Per pbrt-v4 film.h:103, this is `XYZFromSensorRGB`. For
    /// cie1931 with whitebalance=0 it is the identity matrix; for
    /// a named sensor it comes from the 24-swatch least-squares
    /// solve in `build/sensor/build_pixel_sensor.rs`.
    xyz_from_sensor_rgb: [[Float; 3]; 3],
    /// pbrt-v4 `RGBFilm::outputRGBFromSensorRGB` (film.cpp:496)
    /// precomputed as `colorSpace.RGBFromXYZ * sensor.XYZFromSensorRGB`.
    output_rgb_from_sensor_rgb: [[Float; 3]; 3],
}

impl Default for PixelSensor {
    fn default() -> Self {
        Self::create("cie1931", 100.0, 0.0).expect("default pixel sensor should be valid")
    }
}

impl PixelSensor {
    pub fn create(sensor_name: &str, iso: Float, white_balance: Float) -> Result<Self, PbrtError> {
        let sensor_key = sensor_name.to_ascii_lowercase();
        let imaging_ratio = Float::max(0.0, iso) / 100.0;
        match sensor_key.as_str() {
            "cie1931" => {
                // pbrt-v4 `PixelSensor(outputColorSpace, sensorIllum, ...)`
                // (film.h:80-91): cie1931 sensor's r_bar/g_bar/b_bar ARE the
                // CIE X/Y/Z matching functions, so `xyz_from_sensor_rgb`
                // starts as the identity. When the scene asks for a
                // non-default white balance, replace it with a Bradford
                // chromatic adaptation matrix from the D-illuminant at the
                // requested temperature to the output color space's white
                // point (r4 hard-codes sRGB output → D65 white).
                let xyz_from_sensor = if white_balance != 0.0 {
                    let d = d_illuminant(white_balance);
                    let src_white_xy = spectrum_to_xy(&Spectrum::PiecewiseLinear(d));
                    let dst_white_xy = D65_WHITE_XY;
                    bradford_white_balance_matrix(src_white_xy, dst_white_xy)
                } else {
                    IDENTITY_MAT3
                };
                Ok(Self {
                    sensor_name: "cie1931",
                    imaging_ratio,
                    r_bar: DenseSampledSpectrum::from_dense_array(&CIE_X),
                    g_bar: DenseSampledSpectrum::from_dense_array(&CIE_Y),
                    b_bar: DenseSampledSpectrum::from_dense_array(&CIE_Z),
                    xyz_from_sensor_rgb: xyz_from_sensor,
                    output_rgb_from_sensor_rgb: mul_mat3(&srgb_from_xyz_matrix(), &xyz_from_sensor),
                })
            }
            other => {
                let entry = lookup_named_sensor(other).ok_or_else(|| {
                    PbrtError::error(&format!("{}: unknown sensor type", sensor_name))
                })?;
                let r_bar = dense_from_precomputed(entry.r);
                let g_bar = dense_from_precomputed(entry.g);
                let b_bar = dense_from_precomputed(entry.b);
                let xyz_from_sensor = if white_balance != 0.0
                    && Float::abs(white_balance - 6500.0) > 1e-3
                {
                    swatch_lls_xyz_from_sensor(white_balance, &r_bar, &g_bar, &b_bar)
                        .unwrap_or_else(|| {
                            warn!(
                                "swatch LS fit failed for sensor {} whitebalance {}; \
                                 falling back to Bradford adaptation",
                                sensor_name, white_balance
                            );
                            let d_wb = d_illuminant(white_balance);
                            let src_wb_xy = spectrum_to_xy(&Spectrum::PiecewiseLinear(d_wb));
                            let bradford = bradford_white_balance_matrix(src_wb_xy, D65_WHITE_XY);
                            mul_mat3(&bradford, entry.xyz_from_sensor_rgb)
                        })
                } else {
                    *entry.xyz_from_sensor_rgb
                };
                let out_matrix = mul_mat3(&srgb_from_xyz_matrix(), &xyz_from_sensor);
                Ok(Self {
                    sensor_name: entry.name,
                    imaging_ratio,
                    r_bar,
                    g_bar,
                    b_bar,
                    xyz_from_sensor_rgb: xyz_from_sensor,
                    output_rgb_from_sensor_rgb: out_matrix,
                })
            }
        }
    }

    pub fn sensor_name(&self) -> &str {
        self.sensor_name
    }

    pub fn imaging_ratio(&self) -> Float {
        self.imaging_ratio
    }

    pub fn xyz_from_sensor_rgb(&self) -> [[Float; 3]; 3] {
        self.xyz_from_sensor_rgb
    }

    /// pbrt-v4 `PixelSensor::ToSensorRGB` (film.h:95) -- the raw
    /// sensor-space RGB scaled by `imagingRatio`. Callers that
    /// want the output color space's RGB multiply by
    /// `output_rgb_from_sensor_rgb` afterwards (or use the
    /// `to_output_rgb_*` helpers below, which fold both steps).
    pub fn to_sensor_rgb(
        &self,
        spectrum: &SampledSpectrum,
        lambda: &SampledWavelengths,
    ) -> [Float; 3] {
        let r_bar = self.r_bar.sample(lambda);
        let g_bar = self.g_bar.sample(lambda);
        let b_bar = self.b_bar.sample(lambda);
        let radiance = *spectrum / SampledSpectrum::from_pdf(lambda);
        [
            (r_bar * radiance).average() * self.imaging_ratio,
            (g_bar * radiance).average() * self.imaging_ratio,
            (b_bar * radiance).average() * self.imaging_ratio,
        ]
    }

    pub fn to_output_rgb(&self, spectrum: &Spectrum) -> [Float; 3] {
        let spectrum = spectrum.to_dense();
        let sensor_rgb = [
            sampled_inner_product(&self.r_bar, &spectrum),
            sampled_inner_product(&self.g_bar, &spectrum),
            sampled_inner_product(&self.b_bar, &spectrum),
        ];
        let mapped = mul_mat3_vec3(&self.output_rgb_from_sensor_rgb, &sensor_rgb);
        [
            mapped[0] * self.imaging_ratio,
            mapped[1] * self.imaging_ratio,
            mapped[2] * self.imaging_ratio,
        ]
    }

    pub fn to_output_rgb_with_wavelengths(
        &self,
        spectrum: &Spectrum,
        lambda: &SampledWavelengths,
    ) -> [Float; 3] {
        let spectrum = spectrum.to_dense();
        let sensor_rgb = [
            sampled_inner_product_with_wavelengths(&self.r_bar, &spectrum, lambda),
            sampled_inner_product_with_wavelengths(&self.g_bar, &spectrum, lambda),
            sampled_inner_product_with_wavelengths(&self.b_bar, &spectrum, lambda),
        ];
        let mapped = mul_mat3_vec3(&self.output_rgb_from_sensor_rgb, &sensor_rgb);
        [
            mapped[0] * self.imaging_ratio,
            mapped[1] * self.imaging_ratio,
            mapped[2] * self.imaging_ratio,
        ]
    }

    pub fn to_output_rgb_from_packet(
        &self,
        spectrum: &SampledSpectrum,
        lambda: &SampledWavelengths,
    ) -> [Float; 3] {
        let raw_sensor = self.raw_sensor_rgb(spectrum, lambda);
        let mapped = mul_mat3_vec3(&self.output_rgb_from_sensor_rgb, &raw_sensor);
        [
            mapped[0] * self.imaging_ratio,
            mapped[1] * self.imaging_ratio,
            mapped[2] * self.imaging_ratio,
        ]
    }

    /// pbrt-v4 `PixelSensor::ToSensorRGB(L, lambda)` (film.h:97-100):
    /// integrate `r_bar/g_bar/b_bar` against the wavelength-divided
    /// `radiance` and multiply by `imagingRatio`. This stays in
    /// **sensor RGB** (which is CIE XYZ for the `cie1931` sensor) so
    /// that the `maxComponentValue` clamp in `RGBFilm::AddSample`
    /// (film.h:244-247) bites in the same color space as v4 — applying
    /// the clamp downstream in output sRGB would over-clamp because
    /// the `outputRGBFromSensorRGB` matrix has negative entries and
    /// can boost sensor components into output space.
    pub fn to_sensor_rgb_from_packet(
        &self,
        spectrum: &SampledSpectrum,
        lambda: &SampledWavelengths,
    ) -> [Float; 3] {
        let raw_sensor = self.raw_sensor_rgb(spectrum, lambda);
        [
            raw_sensor[0] * self.imaging_ratio,
            raw_sensor[1] * self.imaging_ratio,
            raw_sensor[2] * self.imaging_ratio,
        ]
    }

    /// Apply the `outputRGBFromSensorRGB` colour-space transform to a
    /// sensor-RGB triple (already including `imagingRatio`). Matches
    /// `RGBFilm::GetPixelRGB` (film.h:271) which applies the matrix
    /// at film read-out time rather than per sample.
    pub fn apply_output_matrix(&self, sensor_rgb: &[Float; 3]) -> [Float; 3] {
        mul_mat3_vec3(&self.output_rgb_from_sensor_rgb, sensor_rgb)
    }

    fn raw_sensor_rgb(
        &self,
        spectrum: &SampledSpectrum,
        lambda: &SampledWavelengths,
    ) -> [Float; 3] {
        let r_bar = self.r_bar.sample(lambda);
        let g_bar = self.g_bar.sample(lambda);
        let b_bar = self.b_bar.sample(lambda);
        let radiance = *spectrum / SampledSpectrum::from_pdf(lambda);
        [
            (r_bar * radiance).average(),
            (g_bar * radiance).average(),
            (b_bar * radiance).average(),
        ]
    }
}

const IDENTITY_MAT3: [[Float; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// sRGB output color space's white point chromaticity (CIE D65).
/// Mirrors `RGBColorSpace::sRGB->w` in pbrt-v4.
const D65_WHITE_XY: [Float; 2] = [0.3127, 0.3290];

/// Bradford chromatic adaptation matrices (pbrt-v4
/// `util/color.h:LMSFromXYZ` / `XYZFromLMS`).
const LMS_FROM_XYZ: [[Float; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];
const XYZ_FROM_LMS: [[Float; 3]; 3] = [
    [0.986993, -0.147054, 0.159963],
    [0.432305, 0.51836, 0.0492912],
    [-0.00852866, 0.0400428, 0.968487],
];

/// pbrt-v4 `WhiteBalance(srcWhite, targetWhite)` (`util/color.h`):
/// Bradford-style chromatic adaptation matrix that maps XYZ values
/// referenced to `src_xy` onto XYZ values referenced to `dst_xy`.
/// Used to rebuild `xyz_from_sensor_rgb` when the scene's
/// `whitebalance` differs from the precomputed 6500 K calibration.
fn bradford_white_balance_matrix(src_xy: [Float; 2], dst_xy: [Float; 2]) -> [[Float; 3]; 3] {
    let src_xyz = xyz_from_xy(src_xy);
    let dst_xyz = xyz_from_xy(dst_xy);
    let src_lms = mul_mat3_vec3(&LMS_FROM_XYZ, &src_xyz);
    let dst_lms = mul_mat3_vec3(&LMS_FROM_XYZ, &dst_xyz);
    let lms_correct: [[Float; 3]; 3] = [
        [dst_lms[0] / src_lms[0], 0.0, 0.0],
        [0.0, dst_lms[1] / src_lms[1], 0.0],
        [0.0, 0.0, dst_lms[2] / src_lms[2]],
    ];
    mul_mat3(&XYZ_FROM_LMS, &mul_mat3(&lms_correct, &LMS_FROM_XYZ))
}

/// `XYZ::FromxyY(xy, Y=1)` in pbrt-v4: X = x/y, Y = 1, Z = (1-x-y)/y.
fn xyz_from_xy(xy: [Float; 2]) -> [Float; 3] {
    let x = xy[0];
    let y = xy[1].max(1e-6);
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Integrate a `Spectrum` against the CIE X/Y/Z matching functions
/// to recover its chromaticity (x, y). Used to pull the source white
/// point out of a `d_illuminant(temperature)` when building the
/// Bradford adaptation matrix.
fn spectrum_to_xy(spectrum: &Spectrum) -> [Float; 2] {
    let xyz = spectrum_to_xyz(spectrum);
    let sum = xyz[0] + xyz[1] + xyz[2];
    if sum <= 0.0 {
        return D65_WHITE_XY;
    }
    [xyz[0] / sum, xyz[1] / sum]
}

/// pbrt-v4 `InnerProduct(Spectrum f, Spectrum g)` (util/spectrum.h):
/// 1 nm Riemann sum of `f(λ) * g(λ)` over the visible range. Used to
/// compute `sensorWhiteY` / `sensorWhiteG` for the LS-solve
/// normalisation.
fn inner_product_dense_pl(dense: &DenseSampledSpectrum, pl: &PiecewiseLinearSpectrum) -> Float {
    let mut sum = 0.0;
    let lambda_min = R4_SPECTRUM_LAMBDA_MIN;
    let lambda_max = R4_SPECTRUM_LAMBDA_MAX;
    let mut lambda = lambda_min;
    while lambda <= lambda_max {
        sum += sample_dense_spectrum(dense, lambda) * pl.sample_at(lambda);
        lambda += 1.0;
    }
    sum
}

/// pbrt-v4 `PixelSensor::ProjectReflectance<Triplet>` (film.h:120).
/// 1 nm Riemann sum:
///   result[c] = Σ b_c(λ) · refl(λ) · illum(λ) / Σ b_2(λ) · illum(λ)
fn project_reflectance(
    refl: &PiecewiseLinearSpectrum,
    illum: &PiecewiseLinearSpectrum,
    b1: &DenseSampledSpectrum,
    b2: &DenseSampledSpectrum,
    b3: &DenseSampledSpectrum,
) -> [Float; 3] {
    let mut numer = [0.0; 3];
    let mut denom = 0.0;
    let lambda_min = R4_SPECTRUM_LAMBDA_MIN;
    let lambda_max = R4_SPECTRUM_LAMBDA_MAX;
    let mut lambda = lambda_min;
    while lambda <= lambda_max {
        let r = refl.sample_at(lambda);
        let i = illum.sample_at(lambda);
        let v1 = sample_dense_spectrum(b1, lambda);
        let v2 = sample_dense_spectrum(b2, lambda);
        let v3 = sample_dense_spectrum(b3, lambda);
        numer[0] += v1 * r * i;
        numer[1] += v2 * r * i;
        numer[2] += v3 * r * i;
        denom += v2 * i;
        lambda += 1.0;
    }
    if denom <= 0.0 {
        return [0.0; 3];
    }
    [numer[0] / denom, numer[1] / denom, numer[2] / denom]
}

/// pbrt-v4 `LinearLeastSquares<3>` (util/math.h): solves the normal
/// equations `AᵀA · Mᵀ = AᵀB` for the 3×3 matrix `M` mapping the rows
/// of `a` to the rows of `b` in the least-squares sense, then returns
/// `M` (so `M · a_row^T ≈ b_row^T`).
fn linear_least_squares_3x3(a: &[[Float; 3]], b: &[[Float; 3]]) -> Option<[[Float; 3]; 3]> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut ata = [[0.0_f64; 3]; 3];
    let mut atb = [[0.0_f64; 3]; 3];
    for r in 0..a.len() {
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += a[r][i] as f64 * a[r][j] as f64;
                atb[i][j] += a[r][i] as f64 * b[r][j] as f64;
            }
        }
    }
    let ata_inv = invert_3x3_f64(&ata)?;
    // m^T = ata_inv * atb -> m = transpose(ata_inv * atb)
    let mut mt = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += ata_inv[i][k] * atb[k][j];
            }
            mt[i][j] = s;
        }
    }
    let mut m = [[0.0 as Float; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = mt[j][i] as Float;
        }
    }
    Some(m)
}

fn invert_3x3_f64(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-30 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut inv = [[0.0; 3]; 3];
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    Some(inv)
}

/// pbrt-v4 `PixelSensor(r, g, b, outputColorSpace, sensorIllum, ...)`
/// ctor (film.h:50-79): build `xyz_from_sensor_rgb` by projecting the
/// 24 Macbeth ColorChecker swatches through the sensor's R/G/B curves
/// under the requested whitebalance illuminant, projecting the same
/// swatches through CIE X/Y/Z under the output color space's
/// illuminant (D65 for sRGB), normalising by `sensorWhiteY /
/// sensorWhiteG`, and solving the 3×3 linear least-squares fit.
fn swatch_lls_xyz_from_sensor(
    whitebalance: Float,
    r_bar: &DenseSampledSpectrum,
    g_bar: &DenseSampledSpectrum,
    b_bar: &DenseSampledSpectrum,
) -> Option<[[Float; 3]; 3]> {
    let sensor_illum = d_illuminant(whitebalance);
    // sRGB output color space's illuminant is D65.
    let output_illum = d_illuminant(6500.0);
    let cie_x = DenseSampledSpectrum::from_dense_array(&CIE_X);
    let cie_y = DenseSampledSpectrum::from_dense_array(&CIE_Y);
    let cie_z = DenseSampledSpectrum::from_dense_array(&CIE_Z);

    let sensor_white_g = inner_product_dense_pl(g_bar, &sensor_illum);
    let sensor_white_y = inner_product_dense_pl(&cie_y, &sensor_illum);
    if sensor_white_g <= 0.0 {
        return None;
    }
    let scale = sensor_white_y / sensor_white_g;

    let swatches = swatch_reflectances();
    let mut rgb_camera = Vec::with_capacity(N_SWATCH_REFLECTANCES);
    let mut xyz_output = Vec::with_capacity(N_SWATCH_REFLECTANCES);
    for swatch in swatches.iter() {
        let rgb = project_reflectance(swatch, &sensor_illum, r_bar, g_bar, b_bar);
        rgb_camera.push(rgb);
        let xyz = project_reflectance(swatch, &output_illum, &cie_x, &cie_y, &cie_z);
        xyz_output.push([xyz[0] * scale, xyz[1] * scale, xyz[2] * scale]);
    }
    linear_least_squares_3x3(&rgb_camera, &xyz_output)
}

fn srgb_from_xyz_matrix() -> [[Float; 3]; 3] {
    [
        [3.240479, -1.537150, -0.498535],
        [-0.969256, 1.875991, 0.041556],
        [0.055648, -0.204043, 1.057311],
    ]
}

fn mul_mat3(a: &[[Float; 3]; 3], b: &[[Float; 3]; 3]) -> [[Float; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

fn mul_mat3_vec3(m: &[[Float; 3]; 3], v: &[Float; 3]) -> [Float; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn sampled_inner_product(a: &DenseSampledSpectrum, b: &DenseSampledSpectrum) -> Float {
    // 1nm step over [LAMBDA_MIN, LAMBDA_MAX] -- same convention
    // as `build/sensor/math.rs::inner_product`.
    (0..DenseSampledSpectrum::N_SAMPLES)
        .map(|i| a[i] * b[i])
        .sum::<Float>()
}

fn sampled_inner_product_with_wavelengths(
    a: &DenseSampledSpectrum,
    b: &DenseSampledSpectrum,
    lambda: &SampledWavelengths,
) -> Float {
    let mut sum = 0.0;
    for i in 0..lambda.lambda().len() {
        let pdf = lambda.pdf()[i];
        if pdf == 0.0 {
            continue;
        }

        let aa = sample_dense_spectrum(a, lambda[i]);
        let bb = sample_dense_spectrum(b, lambda[i]);
        sum += (aa * bb) / pdf;
    }

    sum / lambda.lambda().len() as Float
}

fn sample_dense_spectrum(spectrum: &DenseSampledSpectrum, lambda: Float) -> Float {
    let lambda_min = R4_SPECTRUM_LAMBDA_MIN;
    let lambda_max = R4_SPECTRUM_LAMBDA_MAX;
    if lambda < lambda_min || lambda > lambda_max {
        return 0.0;
    }
    let offset = (lambda - lambda_min) / (lambda_max - lambda_min)
        * (DenseSampledSpectrum::N_SAMPLES - 1) as Float;
    let index0 = offset.floor() as usize;
    let index1 = usize::min(index0 + 1, DenseSampledSpectrum::N_SAMPLES - 1);
    let t = offset - index0 as Float;
    lerp(t, spectrum[index0], spectrum[index1])
}

fn dense_from_precomputed(values: &[Float]) -> DenseSampledSpectrum {
    DenseSampledSpectrum::from_dense_array(values)
}
