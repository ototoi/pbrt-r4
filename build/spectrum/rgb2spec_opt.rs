// Numerical building blocks for the Rust port of pbrt-v4's `rgb2spec_opt`.
//
// The generator is enabled only through the explicit
// PBRT_RGB2SPEC_DEBUG_OUTPUT build-time probe until its numerical output has
// been compared with the pbrt-v4 reference.

use super::rgb2spec_data;

const CIE_LAMBDA_MIN: f64 = 360.0;
const CIE_LAMBDA_MAX: f64 = 830.0;
const CIE_SAMPLES: usize = 95;
const CIE_FINE_SAMPLES: usize = (CIE_SAMPLES - 1) * 3 + 1;
const RGB2SPEC_EPSILON: f64 = 1e-4;

#[derive(Clone, Copy)]
pub enum Gamut {
    Srgb,
    Aces2065_1,
    Rec2020,
    DciP3,
}

impl Gamut {
    pub fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("srgb") {
            Some(Self::Srgb)
        } else if name.eq_ignore_ascii_case("aces2065_1") {
            Some(Self::Aces2065_1)
        } else if name.eq_ignore_ascii_case("rec2020") {
            Some(Self::Rec2020)
        } else if name.eq_ignore_ascii_case("dci_p3") {
            Some(Self::DciP3)
        } else {
            None
        }
    }
}

pub struct Tables {
    lambda: [f64; CIE_FINE_SAMPLES],
    rgb: [[f64; CIE_FINE_SAMPLES]; 3],
    rgb_to_xyz: [[f64; 3]; 3],
    xyz_whitepoint: [f64; 3],
}

pub struct GeneratedTable {
    pub scale: Vec<f32>,
    pub coefficients: Vec<f32>,
}

pub fn sigmoid(x: f64) -> f64 {
    0.5 * x / (1.0 + x * x).sqrt() + 0.5
}

pub fn smoothstep(x: f64) -> f64 {
    x * x * (3.0 - 2.0 * x)
}

fn stable_cbrt(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let negative = x < 0.0;
    let ax = x.abs();
    let mut y = 1.0;
    for _ in 0..24 {
        y = (2.0 * y + ax / (y * y)) / 3.0;
    }
    if negative {
        -y
    } else {
        y
    }
}

fn matrix(values: [f64; 9]) -> [[f64; 3]; 3] {
    [
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ]
}

fn cie_interp(data: &[f64; CIE_SAMPLES], x: f64) -> f64 {
    let scaled =
        (x - CIE_LAMBDA_MIN) * (CIE_SAMPLES - 1) as f64 / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);
    let offset = (scaled.floor() as isize).clamp(0, (CIE_SAMPLES - 2) as isize) as usize;
    let weight = scaled - offset as f64;
    (1.0 - weight) * data[offset] + weight * data[offset + 1]
}

impl Tables {
    pub fn new(gamut: Gamut) -> Self {
        let (illuminant, xyz_to_rgb, rgb_to_xyz) = match gamut {
            Gamut::Srgb => (
                &rgb2spec_data::CIE_D65,
                rgb2spec_data::XYZ_TO_SRGB,
                rgb2spec_data::SRGB_TO_XYZ,
            ),
            Gamut::Aces2065_1 => (
                &rgb2spec_data::CIE_D60,
                rgb2spec_data::XYZ_TO_ACES2065_1,
                rgb2spec_data::ACES2065_1_TO_XYZ,
            ),
            Gamut::Rec2020 => (
                &rgb2spec_data::CIE_D65,
                rgb2spec_data::XYZ_TO_REC2020,
                rgb2spec_data::REC2020_TO_XYZ,
            ),
            Gamut::DciP3 => (
                &rgb2spec_data::CIE_D65,
                rgb2spec_data::XYZ_TO_DCI_P3,
                rgb2spec_data::DCI_P3_TO_XYZ,
            ),
        };

        let mut tables = Self {
            lambda: [0.0; CIE_FINE_SAMPLES],
            rgb: [[0.0; CIE_FINE_SAMPLES]; 3],
            rgb_to_xyz: matrix(rgb_to_xyz),
            xyz_whitepoint: [0.0; 3],
        };
        let h = (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / (CIE_FINE_SAMPLES - 1) as f64;
        for i in 0..CIE_FINE_SAMPLES {
            let lambda = CIE_LAMBDA_MIN + i as f64 * h;
            let xyz = [
                cie_interp(&rgb2spec_data::CIE_X, lambda),
                cie_interp(&rgb2spec_data::CIE_Y, lambda),
                cie_interp(&rgb2spec_data::CIE_Z, lambda),
            ];
            let illuminant = cie_interp(illuminant, lambda);
            let mut weight = 3.0 / 8.0 * h;
            if i != 0 && i != CIE_FINE_SAMPLES - 1 {
                weight *= if (i - 1) % 3 == 2 { 2.0 } else { 3.0 };
            }
            tables.lambda[i] = lambda;
            for k in 0..3 {
                for j in 0..3 {
                    tables.rgb[k][i] += xyz_to_rgb[k * 3 + j] * xyz[j] * illuminant * weight;
                }
            }
            for (j, value) in xyz.iter().enumerate() {
                tables.xyz_whitepoint[j] += value * illuminant * weight;
            }
        }
        tables
    }
}

fn cie_lab(tables: &Tables, p: &mut [f64; 3]) {
    let mut xyz = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            xyz[i] += p[j] * tables.rgb_to_xyz[i][j];
        }
    }
    let f = |t: f64| {
        let delta = 6.0 / 29.0;
        if t > delta * delta * delta {
            stable_cbrt(t)
        } else {
            t / (delta * delta * 3.0) + 4.0 / 29.0
        }
    };
    let fx = f(xyz[0] / tables.xyz_whitepoint[0]);
    let fy = f(xyz[1] / tables.xyz_whitepoint[1]);
    let fz = f(xyz[2] / tables.xyz_whitepoint[2]);
    *p = [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)];
}

fn eval_residual(tables: &Tables, coeffs: &[f64; 3], rgb: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for i in 0..CIE_FINE_SAMPLES {
        let lambda = (tables.lambda[i] - CIE_LAMBDA_MIN) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);
        let x = (coeffs[0] * lambda + coeffs[1]) * lambda + coeffs[2];
        let s = sigmoid(x);
        for j in 0..3 {
            out[j] += tables.rgb[j][i] * s;
        }
    }
    cie_lab(tables, &mut out);
    let mut target = *rgb;
    cie_lab(tables, &mut target);
    [target[0] - out[0], target[1] - out[1], target[2] - out[2]]
}

fn eval_jacobian(tables: &Tables, coeffs: &[f64; 3], rgb: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut jac = [[0.0; 3]; 3];
    for i in 0..3 {
        let mut low = *coeffs;
        low[i] -= RGB2SPEC_EPSILON;
        let mut high = *coeffs;
        high[i] += RGB2SPEC_EPSILON;
        let r0 = eval_residual(tables, &low, rgb);
        let r1 = eval_residual(tables, &high, rgb);
        for j in 0..3 {
            jac[j][i] = (r1[j] - r0[j]) / (2.0 * RGB2SPEC_EPSILON);
        }
    }
    jac
}

fn lup_decompose(mut a: [[f64; 3]; 3], tolerance: f64) -> Option<([[f64; 3]; 3], [usize; 4])> {
    let mut p = [0, 1, 2, 3];
    for i in 0..3 {
        let mut imax = i;
        let mut max_a = a[i][i].abs();
        for k in (i + 1)..3 {
            let abs_a = a[k][i].abs();
            if abs_a > max_a {
                max_a = abs_a;
                imax = k;
            }
        }
        if max_a < tolerance {
            return None;
        }
        if imax != i {
            p.swap(i, imax);
            a.swap(i, imax);
            p[3] += 1;
        }
        for j in (i + 1)..3 {
            a[j][i] /= a[i][i];
            for k in (i + 1)..3 {
                a[j][k] -= a[j][i] * a[i][k];
            }
        }
    }
    Some((a, p))
}

fn lup_solve(a: &[[f64; 3]; 3], p: &[usize; 4], b: &[f64; 3]) -> [f64; 3] {
    let mut x = [0.0; 3];
    for i in 0..3 {
        x[i] = b[p[i]];
        for k in 0..i {
            x[i] -= a[i][k] * x[k];
        }
    }
    for i in (0..3).rev() {
        for k in (i + 1)..3 {
            x[i] -= a[i][k] * x[k];
        }
        x[i] /= a[i][i];
    }
    x
}

pub fn gauss_newton(tables: &Tables, rgb: [f64; 3], coeffs: &mut [f64; 3]) {
    // The zero spectrum is a singular limit of the sigmoid-polynomial fit.
    // Keep this case finite and deterministic instead of solving its
    // ill-conditioned Jacobian.
    if rgb == [0.0, 0.0, 0.0] {
        *coeffs = [0.0, 0.0, -1000.0];
        return;
    }

    for _ in 0..15 {
        let residual = eval_residual(tables, coeffs, &rgb);
        let jacobian = eval_jacobian(tables, coeffs, &rgb);
        let Some((decomposed, permutation)) = lup_decompose(jacobian, 1e-15) else {
            panic!("rgb2spec_opt: LU decomposition failed");
        };
        let delta = lup_solve(&decomposed, &permutation, &residual);
        let residual_norm = residual.iter().map(|value| value * value).sum::<f64>();
        for i in 0..3 {
            coeffs[i] -= delta[i];
            // Match the v4 generator's precision boundary between iterations.
            coeffs[i] = coeffs[i] as f32 as f64;
        }
        let max_coeff = coeffs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_coeff > 200.0 {
            for coeff in coeffs.iter_mut() {
                *coeff *= 200.0 / max_coeff;
            }
        }
        if residual_norm < 1e-6 {
            break;
        }
    }
}

fn polynomial_coefficients(coeffs: [f64; 3]) -> [f32; 3] {
    let c0 = 360.0;
    let c1 = 1.0 / (830.0 - 360.0);
    let c1_squared = c1 * c1;
    let c0c1 = c0 * c1;
    let c0c1_squared = c0c1 * c0c1;
    [
        (coeffs[0] * c1_squared) as f32,
        (coeffs[1] * c1 - 2.0 * coeffs[0] * c0 * c1_squared) as f32,
        (coeffs[2] - coeffs[1] * c0c1 + coeffs[0] * c0c1_squared) as f32,
    ]
}

pub fn generate_table(tables: &Tables, resolution: usize) -> GeneratedTable {
    assert!(resolution >= 2);
    let scale: Vec<f32> = (0..resolution)
        .map(|k| smoothstep(smoothstep(k as f64 / (resolution - 1) as f64)) as f32)
        .collect();
    let mut coefficients = vec![0.0_f32; 3 * 3 * resolution * resolution * resolution];

    for l in 0..3 {
        for j in 0..resolution {
            let y = j as f64 / (resolution - 1) as f64;
            for i in 0..resolution {
                let x = i as f64 / (resolution - 1) as f64;
                let mut coeffs = [0.0; 3];
                let start = resolution / 5;

                for k in start..resolution {
                    let b = scale[k] as f64;
                    let mut rgb = [0.0; 3];
                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;
                    gauss_newton(tables, rgb, &mut coeffs);
                    let base = (((l * resolution + k) * resolution + j) * resolution + i) * 3;
                    coefficients[base..base + 3].copy_from_slice(&polynomial_coefficients(coeffs));
                }

                coeffs = [0.0; 3];
                for k in (0..=start).rev() {
                    let b = scale[k] as f64;
                    let mut rgb = [0.0; 3];
                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;
                    gauss_newton(tables, rgb, &mut coeffs);
                    let base = (((l * resolution + k) * resolution + j) * resolution + i) * 3;
                    coefficients[base..base + 3].copy_from_slice(&polynomial_coefficients(coeffs));
                }
            }
        }
    }

    GeneratedTable {
        scale,
        coefficients,
    }
}

pub fn write_table(path: &std::path::Path, table: &GeneratedTable) {
    use std::io::Write;

    let mut output = std::fs::File::create(path).expect("rgb2spec_opt: create output");
    for value in table.scale.iter().chain(table.coefficients.iter()) {
        output
            .write_all(&value.to_le_bytes())
            .expect("rgb2spec_opt: write output");
    }
}

pub fn generate_debug_output() {
    let Ok(path) = std::env::var("PBRT_RGB2SPEC_DEBUG_OUTPUT") else {
        return;
    };
    let gamut_name = std::env::var("PBRT_RGB2SPEC_DEBUG_GAMUT").unwrap_or_else(|_| "sRGB".into());
    let gamut = Gamut::parse(&gamut_name)
        .unwrap_or_else(|| panic!("rgb2spec_opt: unsupported gamut `{gamut_name}`"));
    let resolution = std::env::var("PBRT_RGB2SPEC_DEBUG_RESOLUTION")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    assert!(
        resolution >= 2,
        "rgb2spec_opt: resolution must be at least 2"
    );
    let tables = Tables::new(gamut);
    let generated = generate_table(&tables, resolution);
    write_table(std::path::Path::new(&path), &generated);
}

pub fn validate_constants() {
    assert_eq!(rgb2spec_data::CIE_X.len(), 95);
    assert_eq!(rgb2spec_data::CIE_Y.len(), 95);
    assert_eq!(rgb2spec_data::CIE_Z.len(), 95);
    assert_eq!(rgb2spec_data::CIE_D65.len(), 95);
    assert_eq!(rgb2spec_data::CIE_D60.len(), 95);
    assert_eq!(rgb2spec_data::SRGB_TO_XYZ.len(), 9);
    assert_eq!(rgb2spec_data::XYZ_TO_SRGB.len(), 9);
    assert_eq!(rgb2spec_data::ACES2065_1_TO_XYZ.len(), 9);
    assert_eq!(rgb2spec_data::XYZ_TO_ACES2065_1.len(), 9);
    assert_eq!(rgb2spec_data::REC2020_TO_XYZ.len(), 9);
    assert_eq!(rgb2spec_data::XYZ_TO_REC2020.len(), 9);
    assert_eq!(rgb2spec_data::DCI_P3_TO_XYZ.len(), 9);
    assert_eq!(rgb2spec_data::XYZ_TO_DCI_P3.len(), 9);

    for gamut in [Gamut::Srgb, Gamut::Aces2065_1, Gamut::Rec2020, Gamut::DciP3] {
        let tables = Tables::new(gamut);
        let mut coeffs = [0.0; 3];
        gauss_newton(&tables, [0.5, 0.2, 0.1], &mut coeffs);
        assert!(coeffs.iter().all(|value| value.is_finite()));
        let generated = generate_table(&tables, 2);
        assert_eq!(generated.scale.len(), 2);
        assert_eq!(generated.coefficients.len(), 3 * 3 * 2 * 2 * 2);
        assert!(generated.coefficients.iter().all(|value| value.is_finite()));
    }
}
