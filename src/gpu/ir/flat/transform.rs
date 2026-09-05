pub type Transform = [f32; 16];

pub fn identity_transform() -> Transform {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

pub fn multiply_transform(left: &Transform, right: &Transform) -> Transform {
    let mut result = [0.0; 16];
    for row in 0..4 {
        for column in 0..4 {
            result[row * 4 + column] = (0..4)
                .map(|index| left[row * 4 + index] * right[index * 4 + column])
                .sum();
        }
    }
    result
}

pub fn transform_swaps_handedness(transform: Transform) -> bool {
    let m = transform;
    let determinant = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8])
        + m[2] * (m[4] * m[9] - m[5] * m[8]);
    determinant < 0.0
}

/// Applies the inverse-transpose of the linear part to a normal.
pub fn transform_normal(transform: Transform, normal: [f32; 3]) -> Result<[f32; 3], &'static str> {
    let [a, b, c, _, d, e, f, _, g, h, i, _, _, _, _, _] = transform;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err("transform has a singular linear part");
    }
    let inverse_determinant = 1.0 / determinant;
    let transformed = [
        (e * i - f * h) * normal[0] + (f * g - d * i) * normal[1] + (d * h - e * g) * normal[2],
        (c * h - b * i) * normal[0] + (a * i - c * g) * normal[1] + (b * g - a * h) * normal[2],
        (b * f - c * e) * normal[0] + (c * d - a * f) * normal[1] + (a * e - b * d) * normal[2],
    ];
    let transformed = transformed.map(|value| value * inverse_determinant);
    let length = transformed
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if !length.is_finite() || length == 0.0 {
        return Err("normal becomes zero or non-finite after transformation");
    }
    Ok(transformed.map(|value| value / length))
}
