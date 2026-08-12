//! CPU FLIP image difference metric.
//!
//! The implementation follows pbrt-v4's bundled NVIDIA FLIP reference
//! implementation (`src/ext/flip/flip.cpp`). The license and attribution
//! are retained here because this module is a derived implementation.
//!
//! Derived from the FLIP reference implementation by Pontus Andersson, Jim
//! Nilsson, Tomas Akenine-Moller, Magnus Oskarsson, Kalle Astrom, and Mark D.
//! Fairchild. Copyright (c) 2020, NVIDIA CORPORATION.
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//!
//! * Redistributions of source code must retain the above copyright notice,
//!   this list of conditions and the following disclaimer.
//! * Redistributions in binary form must reproduce the above copyright notice,
//!   this list of conditions and the following disclaimer in the documentation
//!   and/or other materials provided with the distribution.
//! * Neither the name of NVIDIA CORPORATION nor the names of its contributors
//!   may be used to endorse or promote products derived from this software
//!   without specific prior written permission.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
//! EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//! WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
//! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//! LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
//! OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//! DAMAGE.

use crate::util::base::Float;

const QC: Float = 0.7;
const PC: Float = 0.4;
const PT: Float = 0.95;
const GW: Float = 0.082;
const QF: Float = 0.5;
const REF_X: Float = 0.950428545377181;
const REF_Y: Float = 1.0;
const REF_Z: Float = 1.088900370798128;

type Color = [Float; 3];

pub fn error(test: &[Float], reference: &[Float], width: i32, height: i32) -> Vec<Float> {
    let pi = std::f32::consts::PI as Float;
    let ppd: Float = 0.7 * (3840.0 / 0.7) * (pi / 180.0);
    let test = test
        .chunks_exact(3)
        .map(|v| rgb_to_ycxcz([v[0], v[1], v[2]]))
        .collect::<Vec<_>>();
    let reference = reference
        .chunks_exact(3)
        .map(|v| rgb_to_ycxcz([v[0], v[1], v[2]]))
        .collect::<Vec<_>>();

    let filter = spatial_filter(ppd);
    let preprocessed_test = preprocess(&test, width, height, &filter);
    let preprocessed_reference = preprocess(&reference, width, height, &filter);
    let color = color_difference(&preprocessed_reference, &preprocessed_test);
    let feature = feature_difference(&reference, &test, width, height, ppd);
    color
        .into_iter()
        .zip(feature)
        .map(|(color, feature)| (color.powf(1.0 - feature)).clamp(0.0, 1.0))
        .collect()
}

fn srgb_to_linear(value: Float) -> Float {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_xyz(rgb: Color) -> Color {
    [
        10135552.0 / 24577794.0 * rgb[0]
            + 8788810.0 / 24577794.0 * rgb[1]
            + 4435075.0 / 24577794.0 * rgb[2],
        2613072.0 / 12288897.0 * rgb[0]
            + 8788810.0 / 12288897.0 * rgb[1]
            + 887015.0 / 12288897.0 * rgb[2],
        1425312.0 / 73733382.0 * rgb[0]
            + 8788810.0 / 73733382.0 * rgb[1]
            + 70074185.0 / 73733382.0 * rgb[2],
    ]
}

fn xyz_to_linear(xyz: Color) -> Color {
    [
        3.241003232976358 * xyz[0] - 1.537398969488785 * xyz[1] - 0.498615881996363 * xyz[2],
        -0.969224252202516 * xyz[0] + 1.875929983695176 * xyz[1] + 0.041554226340085 * xyz[2],
        0.055639419851975 * xyz[0] - 0.204011206123910 * xyz[1] + 1.057148977187533 * xyz[2],
    ]
}

fn xyz_to_ycxcz(xyz: Color) -> Color {
    let x = xyz[0] / REF_X;
    let y = xyz[1] / REF_Y;
    let z = xyz[2] / REF_Z;
    [116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)]
}

fn ycxcz_to_xyz(value: Color) -> Color {
    let y = (value[0] + 16.0) / 116.0;
    let x = y + value[1] / 500.0;
    let z = y - value[2] / 200.0;
    [x * REF_X, y * REF_Y, z * REF_Z]
}

fn xyz_to_lab(xyz: Color) -> Color {
    let mut v = [
        (xyz[0] / REF_X).abs(),
        (xyz[1] / REF_Y).abs(),
        (xyz[2] / REF_Z).abs(),
    ];
    for component in &mut v {
        *component = if *component > 0.008856 {
            component.powf(1.0 / 3.0)
        } else {
            7.787 * *component + 16.0 / 116.0
        };
    }
    [
        116.0 * v[1] - 16.0,
        500.0 * (v[0] - v[1]),
        200.0 * (v[1] - v[2]),
    ]
}

fn rgb_to_ycxcz(rgb: Color) -> Color {
    xyz_to_ycxcz(linear_to_xyz(rgb.map(srgb_to_linear)))
}

fn hunt(value: Color) -> Color {
    [
        value[0],
        0.01 * value[0] * value[1],
        0.01 * value[0] * value[2],
    ]
}

fn preprocess(image: &[Color], width: i32, height: i32, filter: &[Color]) -> Vec<Color> {
    convolve(image, width, height, filter)
        .into_iter()
        .map(|value| {
            let mut rgb = xyz_to_linear(ycxcz_to_xyz(value));
            rgb.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
            hunt(xyz_to_lab(linear_to_xyz(rgb)))
        })
        .collect()
}

fn gaussian(x: Float, y: Float, sigma: Float) -> Float {
    (-(x * x + y * y) / (2.0 * sigma * sigma)).exp()
}

fn gauss_sum(distance2: Float, a1: Float, b1: Float, a2: Float, b2: Float) -> Float {
    let pi = std::f32::consts::PI as Float;
    a1 * (pi / b1).sqrt() * (-pi * pi * distance2 / b1).exp()
        + a2 * (pi / b2).sqrt() * (-pi * pi * distance2 / b2).exp()
}

fn spatial_filter(ppd: Float) -> Vec<Color> {
    let delta = 1.0 / ppd;
    let a1 = [1.0, 1.0, 34.1];
    let b1 = [0.0047, 0.0053, 0.04];
    let a2 = [0.0, 0.0, 13.5];
    let b2 = [1.0e-5, 1.0e-5, 0.025];
    let max_scale = b1.iter().chain(b2.iter()).copied().fold(0.0, Float::max);
    let pi = std::f32::consts::PI as Float;
    let radius = (3.0 * (max_scale / (2.0 * pi.powi(2))).sqrt() * ppd).ceil() as i32;
    let width = 2 * radius + 1;
    let mut filter = Vec::with_capacity((width * width) as usize);
    let mut sum = [0.0; 3];
    for y in -radius..=radius {
        for x in -radius..=radius {
            let distance2 = (x as Float * delta).powi(2) + (y as Float * delta).powi(2);
            let value = [
                gauss_sum(distance2, a1[0], b1[0], a2[0], b2[0]),
                gauss_sum(distance2, a1[1], b1[1], a2[1], b2[1]),
                gauss_sum(distance2, a1[2], b1[2], a2[2], b2[2]),
            ];
            sum = add(sum, value);
            filter.push(value);
        }
    }
    filter.into_iter().map(|value| div(value, sum)).collect()
}

fn detection_filter(ppd: Float, point: bool) -> Vec<Color> {
    let sigma = 0.5 * GW * ppd;
    let radius = (3.0 * sigma).ceil() as i32;
    let mut filter = Vec::new();
    let mut positive = [0.0; 2];
    let mut negative = [0.0; 2];
    for y in -radius..=radius {
        for x in -radius..=radius {
            let g = gaussian(x as Float, y as Float, sigma);
            let wx = if point {
                (x as Float * x as Float / (sigma * sigma) - 1.0) * g
            } else {
                -(x as Float) * g
            };
            let wy = if point {
                (y as Float * y as Float / (sigma * sigma) - 1.0) * g
            } else {
                -(y as Float) * g
            };
            if wx > 0.0 {
                positive[0] += wx
            } else {
                negative[0] += wx
            }
            if wy > 0.0 {
                positive[1] += wy
            } else {
                negative[1] += wy
            }
            filter.push([wx, wy, 0.0]);
        }
    }
    filter
        .into_iter()
        .map(|value| {
            [
                value[0]
                    / if value[0] > 0.0 {
                        positive[0]
                    } else {
                        -negative[0]
                    },
                value[1]
                    / if value[1] > 0.0 {
                        positive[1]
                    } else {
                        -negative[1]
                    },
                0.0,
            ]
        })
        .collect()
}

fn convolve(image: &[Color], width: i32, height: i32, filter: &[Color]) -> Vec<Color> {
    let side = (filter.len() as f32).sqrt() as i32;
    let radius = side / 2;
    let mut output = vec![[0.0; 3]; image.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = [0.0; 3];
            for fy in 0..side {
                for fx in 0..side {
                    let sx = (x + fx - radius).clamp(0, width - 1);
                    let sy = (y + fy - radius).clamp(0, height - 1);
                    let source = image[(sy * width + sx) as usize];
                    value = add(value, mul(source, filter[(fy * side + fx) as usize]));
                }
            }
            output[(y * width + x) as usize] = value;
        }
    }
    output
}

fn color_difference(reference: &[Color], test: &[Color]) -> Vec<Float> {
    let green = hunt(xyz_to_lab(linear_to_xyz([0.0, 1.0, 0.0])));
    let blue = hunt(xyz_to_lab(linear_to_xyz([0.0, 0.0, 1.0])));
    let cmax = hyab(green, blue).powf(QC);
    let pccmax = PC * cmax;
    reference
        .iter()
        .zip(test)
        .map(|(&a, &b)| {
            let mut error = hyab(a, b).powf(QC);
            if error < pccmax {
                error *= PT / pccmax;
            } else {
                error = PT + (error - pccmax) / (cmax - pccmax) * (1.0 - PT);
            }
            error
        })
        .collect()
}

fn feature_difference(
    reference: &[Color],
    test: &[Color],
    width: i32,
    height: i32,
    ppd: Float,
) -> Vec<Float> {
    let gray = |image: &[Color]| {
        image
            .iter()
            .map(|value| {
                let c = (value[0] + 16.0) / 116.0;
                [c, c, 0.0]
            })
            .collect::<Vec<_>>()
    };
    let ref_gray = gray(reference);
    let test_gray = gray(test);
    let edge_filter = detection_filter(ppd, false);
    let point_filter = detection_filter(ppd, true);
    let ref_edges = convolve(&ref_gray, width, height, &edge_filter);
    let test_edges = convolve(&test_gray, width, height, &edge_filter);
    let ref_points = convolve(&ref_gray, width, height, &point_filter);
    let test_points = convolve(&test_gray, width, height, &point_filter);
    let scale = (2.0 as Float).sqrt().recip();
    ref_edges
        .iter()
        .zip(test_edges)
        .zip(ref_points.iter().zip(test_points))
        .map(|((&re, te), (&rp, tp))| {
            let edge = (re[0] * re[0] + re[1] * re[1]).sqrt();
            let edge_test = (te[0] * te[0] + te[1] * te[1]).sqrt();
            let point = (rp[0] * rp[0] + rp[1] * rp[1]).sqrt();
            let point_test = (tp[0] * tp[0] + tp[1] * tp[1]).sqrt();
            (scale * (edge - edge_test).abs().max((point - point_test).abs())).powf(QF)
        })
        .collect()
}

fn hyab(a: Color, b: Color) -> Float {
    (a[0] - b[0]).abs() + ((a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn add(a: Color, b: Color) -> Color {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: Color, b: Color) -> Color {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}
fn div(a: Color, b: Color) -> Color {
    [a[0] / b[0], a[1] / b[1], a[2] / b[2]]
}
