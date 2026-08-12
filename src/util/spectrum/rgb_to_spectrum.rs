use std::sync::OnceLock;

use crate::util::base::Float;

use super::config::{LAMBDA_MAX, LAMBDA_MIN};
use super::densely_sampled::DenselySampledSpectrum;
use super::named_arrays::{ACES_Illum_D60, CIE_Illum_D6500};
use super::sampled::{SampledSpectrum, SampledWavelengths};
use super::source::SpectrumType;

include!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_meta.rs"));

const SRGB_TO_SPECTRUM_BIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_srgb.bin"));

const ACES_TO_SPECTRUM_BIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_aces.bin"));

const DCI_P3_TO_SPECTRUM_BIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_dci_p3.bin"));

const REC2020_TO_SPECTRUM_BIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_rec2020.bin"));

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RGBSigmoidPolynomial {
    c0: Float,
    c1: Float,
    c2: Float,
}

impl RGBSigmoidPolynomial {
    pub fn new(c0: Float, c1: Float, c2: Float) -> Self {
        Self { c0, c1, c2 }
    }

    pub fn eval(&self, lambda: Float) -> Float {
        let x = ((self.c0 * lambda + self.c1) * lambda) + self.c2;
        sigmoid(x)
    }

    pub fn max_value(&self) -> Float {
        let mut result = self
            .eval(LAMBDA_MIN as Float)
            .max(self.eval(LAMBDA_MAX as Float));
        if self.c0 != 0.0 {
            let lambda = -self.c1 / (2.0 * self.c0);
            if (LAMBDA_MIN as Float..=LAMBDA_MAX as Float).contains(&lambda) {
                result = result.max(self.eval(lambda));
            }
        }
        result
    }
}

struct RGBToSpectrumTableData {
    floats: Box<[f32]>,
}

impl RGBToSpectrumTableData {
    fn decode() -> Self {
        Self::decode_bytes(SRGB_TO_SPECTRUM_BIN)
    }

    fn decode_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len() % std::mem::size_of::<f32>(), 0);

        let mut floats = Vec::with_capacity(bytes.len() / std::mem::size_of::<f32>());
        for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        // The bin layout matches pbrt-v4's generated table:
        //   [z scale (64)] [coeffs (3 * 64 * 64 * 64 * 3)]
        let expected_len = SRGB_TO_SPECTRUM_SCALE_COUNT + SRGB_TO_SPECTRUM_COEFF_COUNT;
        assert_eq!(floats.len(), expected_len);

        Self {
            floats: floats.into_boxed_slice(),
        }
    }

    fn z_nodes(&self) -> &[f32] {
        &self.floats[..SRGB_TO_SPECTRUM_SCALE_COUNT]
    }

    fn coeffs(&self) -> &[f32] {
        let start = SRGB_TO_SPECTRUM_SCALE_COUNT;
        &self.floats[start..]
    }

    fn coeff(&self, maxc: usize, z: usize, y: usize, x: usize, component: usize) -> Float {
        let index = (((((maxc * SRGB_TO_SPECTRUM_RES) + z) * SRGB_TO_SPECTRUM_RES + y)
            * SRGB_TO_SPECTRUM_RES
            + x)
            * 3)
            + component;
        self.coeffs()[index] as Float
    }
}

fn sample_named_illuminant(data: &[Float; 214], lambda: Float) -> Float {
    let count = data.len() / 2;
    if lambda <= data[0] {
        return data[1];
    }
    if lambda >= data[2 * (count - 1)] {
        return data[2 * (count - 1) + 1];
    }

    let mut low = 0usize;
    let mut high = count - 1;
    while high - low > 1 {
        let mid = (low + high) / 2;
        if data[2 * mid] <= lambda {
            low = mid;
        } else {
            high = mid;
        }
    }
    let lambda0 = data[2 * low];
    let lambda1 = data[2 * high];
    let value0 = data[2 * low + 1];
    let value1 = data[2 * high + 1];
    let t = (lambda - lambda0) / (lambda1 - lambda0);
    (1.0 - t) * value0 + t * value1
}

fn srgb_table_data() -> &'static RGBToSpectrumTableData {
    static TABLE: OnceLock<RGBToSpectrumTableData> = OnceLock::new();
    TABLE.get_or_init(RGBToSpectrumTableData::decode)
}

fn aces_table_data() -> &'static RGBToSpectrumTableData {
    static TABLE: OnceLock<RGBToSpectrumTableData> = OnceLock::new();
    TABLE.get_or_init(|| RGBToSpectrumTableData::decode_bytes(ACES_TO_SPECTRUM_BIN))
}

fn dci_p3_table_data() -> &'static RGBToSpectrumTableData {
    static TABLE: OnceLock<RGBToSpectrumTableData> = OnceLock::new();
    TABLE.get_or_init(|| RGBToSpectrumTableData::decode_bytes(DCI_P3_TO_SPECTRUM_BIN))
}

fn rec2020_table_data() -> &'static RGBToSpectrumTableData {
    static TABLE: OnceLock<RGBToSpectrumTableData> = OnceLock::new();
    TABLE.get_or_init(|| RGBToSpectrumTableData::decode_bytes(REC2020_TO_SPECTRUM_BIN))
}

fn clamp_zero_rgb(rgb: [Float; 3]) -> [Float; 3] {
    [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]
}

fn clamp_albedo_rgb(rgb: [Float; 3]) -> [Float; 3] {
    [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ]
}

fn sigmoid(x: Float) -> Float {
    if x.is_infinite() {
        return if x.is_sign_positive() { 1.0 } else { 0.0 };
    }
    0.5 + x / (2.0 * (1.0 + x * x).sqrt())
}

fn find_z_interval(z_nodes: &[f32], z: Float) -> usize {
    if z <= z_nodes[0] as Float {
        return 0;
    }
    let mut zi = 0usize;
    while zi + 1 < z_nodes.len() - 1 && (z_nodes[zi + 1] as Float) < z {
        zi += 1;
    }
    zi
}

fn lookup_srgb_sigmoid(rgb: [Float; 3]) -> RGBSigmoidPolynomial {
    lookup_sigmoid(rgb, srgb_table_data())
}

/// Generalised RGB → sigmoid polynomial lookup. The table layout is
/// identical across colour spaces (sRGB / ACES2065-1 / ... share the
/// `[3][res][res][res][3]` coefficient grid + `[res]` z-scale shape),
/// so the only difference between colour spaces is which precomputed
/// table is used.
fn lookup_sigmoid(rgb: [Float; 3], table: &RGBToSpectrumTableData) -> RGBSigmoidPolynomial {
    debug_assert!(rgb
        .iter()
        .all(|component| *component >= 0.0 && *component <= 1.0));

    if rgb[0] == rgb[1] && rgb[1] == rgb[2] {
        let c2 = (rgb[0] - 0.5) / (rgb[0] * (1.0 - rgb[0])).sqrt();
        return RGBSigmoidPolynomial::new(0.0, 0.0, c2);
    }

    let maxc = if rgb[0] > rgb[1] {
        if rgb[0] > rgb[2] {
            0
        } else {
            2
        }
    } else if rgb[1] > rgb[2] {
        1
    } else {
        2
    };

    let z = rgb[maxc];
    let x = rgb[(maxc + 1) % 3] * (SRGB_TO_SPECTRUM_RES as Float - 1.0) / z;
    let y = rgb[(maxc + 2) % 3] * (SRGB_TO_SPECTRUM_RES as Float - 1.0) / z;

    let xi = usize::min(x.floor() as usize, SRGB_TO_SPECTRUM_RES - 2);
    let yi = usize::min(y.floor() as usize, SRGB_TO_SPECTRUM_RES - 2);

    let zi = find_z_interval(table.z_nodes(), z);
    let dx = x - xi as Float;
    let dy = y - yi as Float;
    let z0 = table.z_nodes()[zi] as Float;
    let z1 = table.z_nodes()[zi + 1] as Float;
    let dz = (z - z0) / (z1 - z0);

    let mut c = [0.0; 3];
    for (component, coefficient) in c.iter_mut().enumerate() {
        let c000 = table.coeff(maxc, zi, yi, xi, component);
        let c100 = table.coeff(maxc, zi, yi, xi + 1, component);
        let c010 = table.coeff(maxc, zi, yi + 1, xi, component);
        let c110 = table.coeff(maxc, zi, yi + 1, xi + 1, component);
        let c001 = table.coeff(maxc, zi + 1, yi, xi, component);
        let c101 = table.coeff(maxc, zi + 1, yi, xi + 1, component);
        let c011 = table.coeff(maxc, zi + 1, yi + 1, xi, component);
        let c111 = table.coeff(maxc, zi + 1, yi + 1, xi + 1, component);

        let c00 = (1.0 - dx) * c000 + dx * c100;
        let c10 = (1.0 - dx) * c010 + dx * c110;
        let c01 = (1.0 - dx) * c001 + dx * c101;
        let c11 = (1.0 - dx) * c011 + dx * c111;
        let c0 = (1.0 - dy) * c00 + dy * c10;
        let c1 = (1.0 - dy) * c01 + dy * c11;
        *coefficient = (1.0 - dz) * c0 + dz * c1;
    }

    RGBSigmoidPolynomial::new(c[0], c[1], c[2])
}

/// Normalize the D65 illuminant by `CIE_Y_INTEGRAL / ∫ Y(λ) · D65_raw(λ) dλ`,
/// matching pbrt-v4's `RGBColorSpace::illuminant.Scale(CIE_Y_integral / y_integral)`.
/// Without this, RGB-illuminant-driven emissive lights come out ~99× too bright
/// (= raw D65's photometric integral / CIE_Y_INTEGRAL).
fn d65_normalization_factor() -> Float {
    static FACTOR: OnceLock<Float> = OnceLock::new();
    *FACTOR.get_or_init(|| {
        use super::cie::{CIE_LAMBDA, CIE_SAMPLES, CIE_Y, CIE_Y_INTEGRAL};
        // Riemann-like integration of `D65_raw(λ) · Y(λ)` over the CIE λ range.
        let mut integral = 0.0;
        for i in 0..CIE_SAMPLES {
            let lam = CIE_LAMBDA[i];
            let d65 = sample_named_illuminant(&CIE_Illum_D6500, lam);
            integral += d65 * CIE_Y[i];
        }
        let step = CIE_LAMBDA[1] - CIE_LAMBDA[0]; // assumed uniform 1nm
        let y_integral = integral * step;
        CIE_Y_INTEGRAL / y_integral
    })
}

pub fn d65_sample(lambda: Float) -> Float {
    let raw = sample_named_illuminant(&CIE_Illum_D6500, lambda);
    raw * d65_normalization_factor()
}

pub fn d65_max_value() -> Float {
    static MAX_VALUE: OnceLock<Float> = OnceLock::new();
    *MAX_VALUE.get_or_init(|| {
        let raw = CIE_Illum_D6500
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .fold(0.0, Float::max);
        raw * d65_normalization_factor()
    })
}

pub fn srgb_albedo_to_polynomial(rgb: [Float; 3]) -> RGBSigmoidPolynomial {
    lookup_srgb_sigmoid(clamp_albedo_rgb(rgb))
}

pub fn srgb_unbounded_to_scaled_polynomial(rgb: [Float; 3]) -> (Float, RGBSigmoidPolynomial) {
    let rgb = clamp_zero_rgb(rgb);
    let max_component = rgb[0].max(rgb[1]).max(rgb[2]);
    let scale = 2.0 * max_component;
    let normalized = if scale > 0.0 {
        [rgb[0] / scale, rgb[1] / scale, rgb[2] / scale]
    } else {
        [0.0, 0.0, 0.0]
    };
    (scale, lookup_srgb_sigmoid(normalized))
}

pub fn srgb_illuminant_to_scaled_polynomial(rgb: [Float; 3]) -> (Float, RGBSigmoidPolynomial) {
    srgb_unbounded_to_scaled_polynomial(rgb)
}

pub fn srgb_albedo_to_dense_spectrum(rgb: [Float; 3]) -> DenselySampledSpectrum {
    let polynomial = srgb_albedo_to_polynomial(rgb);
    DenselySampledSpectrum::sample_function(|lambda| polynomial.eval(lambda))
}

pub fn srgb_unbounded_to_dense_spectrum(rgb: [Float; 3]) -> DenselySampledSpectrum {
    let (scale, polynomial) = srgb_unbounded_to_scaled_polynomial(rgb);
    DenselySampledSpectrum::sample_function(|lambda| scale * polynomial.eval(lambda))
}

pub fn srgb_illuminant_to_dense_spectrum(rgb: [Float; 3]) -> DenselySampledSpectrum {
    let (scale, polynomial) = srgb_illuminant_to_scaled_polynomial(rgb);
    DenselySampledSpectrum::sample_function(|lambda| {
        scale * polynomial.eval(lambda) * d65_sample(lambda)
    })
}

pub fn srgb_albedo_to_sampled_spectrum(
    rgb: [Float; 3],
    lambda: &SampledWavelengths,
) -> SampledSpectrum {
    let polynomial = srgb_albedo_to_polynomial(rgb);
    let mut values = [0.0; SampledSpectrum::N_SAMPLES];
    for i in 0..SampledSpectrum::N_SAMPLES {
        values[i] = polynomial.eval(lambda[i]).max(0.0);
    }
    SampledSpectrum::from(values)
}

pub fn srgb_unbounded_to_sampled_spectrum(
    rgb: [Float; 3],
    lambda: &SampledWavelengths,
) -> SampledSpectrum {
    let (scale, polynomial) = srgb_unbounded_to_scaled_polynomial(rgb);
    let mut values = [0.0; SampledSpectrum::N_SAMPLES];
    for i in 0..SampledSpectrum::N_SAMPLES {
        values[i] = (scale * polynomial.eval(lambda[i])).max(0.0);
    }
    SampledSpectrum::from(values)
}

pub fn srgb_illuminant_to_sampled_spectrum(
    rgb: [Float; 3],
    lambda: &SampledWavelengths,
) -> SampledSpectrum {
    let (scale, polynomial) = srgb_illuminant_to_scaled_polynomial(rgb);
    let mut values = [0.0; SampledSpectrum::N_SAMPLES];
    for i in 0..SampledSpectrum::N_SAMPLES {
        values[i] = (scale * polynomial.eval(lambda[i]) * d65_sample(lambda[i])).max(0.0);
    }
    SampledSpectrum::from(values)
}

pub fn srgb_to_sampled_spectrum(
    rgb: [Float; 3],
    spectrum_type: SpectrumType,
    lambda: &SampledWavelengths,
) -> SampledSpectrum {
    match spectrum_type {
        SpectrumType::Albedo => srgb_albedo_to_sampled_spectrum(rgb, lambda),
        SpectrumType::Unbounded => srgb_unbounded_to_sampled_spectrum(rgb, lambda),
        SpectrumType::Illuminant => srgb_illuminant_to_sampled_spectrum(rgb, lambda),
    }
}

// ============================================================================
// Multi-colour-space support
// ============================================================================
//
// pbrt-v4 keeps each `RGBColorSpace` (sRGB / ACES2065-1 / DCI-P3 /
// Rec2020) with its own precomputed RGB→spectrum table plus the
// associated whitepoint illuminant (D65 for sRGB, D60 for ACES, ...).
// `RGBIlluminantSpectrum(*imageColorSpace, rgb)` reads from that
// colour space's table; the illuminant lookup likewise comes from
// `cs.illuminant`.
//
// r4 carries the named colour spaces initialized by pbrt-v4:
// sRGB, ACES2065-1, DCI-P3, and Rec2020.

/// Statically initializable equivalent of pbrt-v4's
/// `RGBColorSpace::illuminant` densely sampled spectrum.
#[derive(Debug, Clone, Copy)]
pub struct RGBColorSpaceIlluminant {
    sample: fn(Float) -> Float,
}

impl PartialEq for RGBColorSpaceIlluminant {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.sample, other.sample)
    }
}

impl RGBColorSpaceIlluminant {
    pub const fn new(sample: fn(Float) -> Float) -> Self {
        Self { sample }
    }

    pub fn sample_at(&self, lambda: Float) -> Float {
        (self.sample)(lambda)
    }

    pub fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut values = [0.0; SampledSpectrum::N_SAMPLES];
        for i in 0..SampledSpectrum::N_SAMPLES {
            values[i] = self.sample_at(lambda[i]);
        }
        SampledSpectrum::from(values)
    }

    pub fn to_dense(&self) -> DenselySampledSpectrum {
        DenselySampledSpectrum::sample_function(|lambda| self.sample_at(lambda))
    }

    pub fn max_value(&self) -> Float {
        self.to_dense().max_value()
    }
}

/// pbrt-v4 `class RGBColorSpace` (util/colorspace.h). r4 stores the
/// fields needed for CPU rendering: chromaticities (for `Lookup`
/// matching against EXR metadata), the colour-space illuminant, and
/// the per-colour-space sigmoid-polynomial table.
#[derive(Clone, Copy)]
pub struct RGBColorSpace {
    pub name: &'static str,
    pub r: [Float; 2],
    pub g: [Float; 2],
    pub b: [Float; 2],
    pub w: [Float; 2],
    pub illuminant: RGBColorSpaceIlluminant,
    table: fn() -> &'static RGBToSpectrumTableData,
}

impl RGBColorSpace {
    /// Convert CIE XYZ values to this color space, matching v4's
    /// `RGBColorSpace::ToRGB`. The matrix is derived from this space's
    /// primaries and white point rather than using the fixed sRGB matrix.
    pub fn xyz_to_rgb(&self, xyz: [Float; 3]) -> [Float; 3] {
        let primary_xyz = |primary: [Float; 2]| {
            [
                primary[0] / primary[1],
                1.0,
                (1.0 - primary[0] - primary[1]) / primary[1],
            ]
        };
        let r = primary_xyz(self.r);
        let g = primary_xyz(self.g);
        let b = primary_xyz(self.b);
        let white = primary_xyz(self.w);
        let matrix = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
        let inverse = invert_3x3(matrix);
        let scale = [
            inverse[0][0] * white[0] + inverse[0][1] * white[1] + inverse[0][2] * white[2],
            inverse[1][0] * white[0] + inverse[1][1] * white[1] + inverse[1][2] * white[2],
            inverse[2][0] * white[0] + inverse[2][1] * white[1] + inverse[2][2] * white[2],
        ];
        let xyz_from_rgb = [
            [
                matrix[0][0] * scale[0],
                matrix[0][1] * scale[1],
                matrix[0][2] * scale[2],
            ],
            [
                matrix[1][0] * scale[0],
                matrix[1][1] * scale[1],
                matrix[1][2] * scale[2],
            ],
            [
                matrix[2][0] * scale[0],
                matrix[2][1] * scale[1],
                matrix[2][2] * scale[2],
            ],
        ];
        let inverse = invert_3x3(xyz_from_rgb);
        [
            inverse[0][0] * xyz[0] + inverse[0][1] * xyz[1] + inverse[0][2] * xyz[2],
            inverse[1][0] * xyz[0] + inverse[1][1] * xyz[1] + inverse[1][2] * xyz[2],
            inverse[2][0] * xyz[0] + inverse[2][1] * xyz[1] + inverse[2][2] * xyz[2],
        ]
    }

    pub fn albedo_to_polynomial(&self, rgb: [Float; 3]) -> RGBSigmoidPolynomial {
        lookup_sigmoid(clamp_albedo_rgb(rgb), (self.table)())
    }

    pub fn unbounded_to_scaled_polynomial(&self, rgb: [Float; 3]) -> (Float, RGBSigmoidPolynomial) {
        let rgb = clamp_zero_rgb(rgb);
        let max_component = rgb[0].max(rgb[1]).max(rgb[2]);
        let scale = 2.0 * max_component;
        let normalized = if scale > 0.0 {
            [rgb[0] / scale, rgb[1] / scale, rgb[2] / scale]
        } else {
            [0.0, 0.0, 0.0]
        };
        (scale, lookup_sigmoid(normalized, (self.table)()))
    }

    pub fn illuminant_to_scaled_polynomial(
        &self,
        rgb: [Float; 3],
    ) -> (Float, RGBSigmoidPolynomial) {
        self.unbounded_to_scaled_polynomial(rgb)
    }

    /// pbrt-v4 `RGBIlluminantSpectrum(cs, rgb).Sample(lambda)`
    /// (spectrum.cpp:RGBIlluminantSpectrum + spectrum.h:operator()):
    /// `scale * rsp(λ) * cs.illuminant(λ)`. The polynomial comes from
    /// the colour-space-specific table and the illuminant from the
    /// colour-space-specific whitepoint sampler.
    pub fn illuminant_to_sampled_spectrum(
        &self,
        rgb: [Float; 3],
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        let (scale, polynomial) = self.illuminant_to_scaled_polynomial(rgb);
        let mut values = [0.0; SampledSpectrum::N_SAMPLES];
        for i in 0..SampledSpectrum::N_SAMPLES {
            values[i] = (scale * polynomial.eval(lambda[i]) * self.illuminant.sample_at(lambda[i]))
                .max(0.0);
        }
        SampledSpectrum::from(values)
    }
}

fn invert_3x3(matrix: [[Float; 3]; 3]) -> [[Float; 3]; 3] {
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    assert!(determinant != 0.0, "RGB color space matrix is singular");
    [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) / determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / determinant,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) / determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) / determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) / determinant,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) / determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) / determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / determinant,
        ],
    ]
}

pub static SRGB: RGBColorSpace = RGBColorSpace {
    name: "sRGB",
    r: [0.64, 0.33],
    g: [0.30, 0.60],
    b: [0.15, 0.06],
    w: [0.3127, 0.3290],
    illuminant: RGBColorSpaceIlluminant::new(d65_sample),
    table: srgb_table_data,
};

pub static ACES2065_1: RGBColorSpace = RGBColorSpace {
    name: "ACES2065-1",
    r: [0.7347, 0.2653],
    g: [0.0000, 1.0000],
    b: [0.0001, -0.0770],
    w: [0.32168, 0.33767],
    illuminant: RGBColorSpaceIlluminant::new(aces_d60_sample),
    table: aces_table_data,
};

// pbrt-v4 `RGBColorSpace::DCI_P3` (colorspace.cpp:86-88). DCI-P3
// display primaries with the **D65 whitepoint** (used by EXR sources
// produced for HDR consumer/display delivery).
pub static DCI_P3: RGBColorSpace = RGBColorSpace {
    name: "DCI-P3",
    r: [0.68, 0.32],
    g: [0.265, 0.690],
    b: [0.15, 0.06],
    w: [0.3127, 0.3290],
    illuminant: RGBColorSpaceIlluminant::new(d65_sample),
    table: dci_p3_table_data,
};

// pbrt-v4 `RGBColorSpace::Rec2020` (colorspace.cpp:89-91). ITU-R
// BT.2020 primaries, also D65 white. Wide gamut used by UHD HDR
// pipelines.
pub static REC2020: RGBColorSpace = RGBColorSpace {
    name: "Rec2020",
    r: [0.708, 0.292],
    g: [0.170, 0.797],
    b: [0.131, 0.046],
    w: [0.3127, 0.3290],
    illuminant: RGBColorSpaceIlluminant::new(d65_sample),
    table: rec2020_table_data,
};

/// pbrt-v4 `RGBColorSpace::GetNamed(name)` (util/colorspace.cpp:80-95).
/// Maps the names accepted by the `ColorSpace` scene directive to the
/// shipped color spaces. Returns `None` for unknown names.
pub fn lookup_color_space_by_name(name: &str) -> Option<&'static RGBColorSpace> {
    match name {
        "srgb" => Some(&SRGB),
        "aces2065-1" => Some(&ACES2065_1),
        "rec2020" => Some(&REC2020),
        "dci-p3" => Some(&DCI_P3),
        _ => None,
    }
}

/// pbrt-v4 `RGBColorSpace::Lookup(r, g, b, w)`
/// (util/colorspace.cpp:96-108). Returns the matching named colour
/// space or `None` if no candidate is within `1e-3` relative tolerance
/// on every chromaticity.
pub fn lookup_color_space(
    r: [Float; 2],
    g: [Float; 2],
    b: [Float; 2],
    w: [Float; 2],
) -> Option<&'static RGBColorSpace> {
    fn close_enough(a: [Float; 2], b: [Float; 2]) -> bool {
        let rel = |x: Float, y: Float| (x - y).abs() / y.abs().max(Float::EPSILON);
        (a[0] == b[0] || rel(a[0], b[0]) < 1e-3) && (a[1] == b[1] || rel(a[1], b[1]) < 1e-3)
    }
    // pbrt-v4 tests ACES2065_1 / DCI_P3 / Rec2020 / sRGB in that
    // order (colorspace.cpp:103). r4 mirrors the order.
    for cs in [&ACES2065_1, &DCI_P3, &REC2020, &SRGB] {
        if close_enough(r, cs.r)
            && close_enough(g, cs.g)
            && close_enough(b, cs.b)
            && close_enough(w, cs.w)
        {
            return Some(cs);
        }
    }
    None
}

/// ACES D60 illuminant sample, normalised by
/// `CIE_Y_INTEGRAL / ∫ Y(λ) · D60_raw(λ) dλ`. Same scheme as
/// `d65_sample` so `RGBIlluminantSpectrum`-style scaling stays
/// consistent across colour spaces.
pub fn aces_d60_sample(lambda: Float) -> Float {
    let raw = sample_named_illuminant(&ACES_Illum_D60, lambda);
    raw * aces_d60_normalization_factor()
}

fn aces_d60_normalization_factor() -> Float {
    static FACTOR: OnceLock<Float> = OnceLock::new();
    *FACTOR.get_or_init(|| {
        use super::cie::{CIE_LAMBDA, CIE_SAMPLES, CIE_Y, CIE_Y_INTEGRAL};
        let mut integral = 0.0;
        for i in 0..CIE_SAMPLES {
            let lam = CIE_LAMBDA[i];
            let d = sample_named_illuminant(&ACES_Illum_D60, lam);
            integral += d * CIE_Y[i];
        }
        let step = CIE_LAMBDA[1] - CIE_LAMBDA[0]; // assumed uniform 1nm
        let y_integral = integral * step;
        CIE_Y_INTEGRAL / y_integral
    })
}
