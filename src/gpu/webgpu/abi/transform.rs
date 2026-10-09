use crate::util::error::PbrtError;

pub fn inverse_affine(transform: [f32; 16], label: &str) -> Result<[[f32; 4]; 4], PbrtError> {
    let [a, b, c, tx, d, e, f, ty, g, h, i, tz, _, _, _, _] = transform;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(PbrtError::error(&format!(
            "{label} transform has a singular linear part."
        )));
    }
    let inverse_determinant = 1.0 / determinant;
    let inverse_linear = [
        (e * i - f * h) * inverse_determinant,
        (c * h - b * i) * inverse_determinant,
        (b * f - c * e) * inverse_determinant,
        (f * g - d * i) * inverse_determinant,
        (a * i - c * g) * inverse_determinant,
        (c * d - a * f) * inverse_determinant,
        (d * h - e * g) * inverse_determinant,
        (b * g - a * h) * inverse_determinant,
        (a * e - b * d) * inverse_determinant,
    ];
    let inverse_translation = [
        -(inverse_linear[0] * tx + inverse_linear[1] * ty + inverse_linear[2] * tz),
        -(inverse_linear[3] * tx + inverse_linear[4] * ty + inverse_linear[5] * tz),
        -(inverse_linear[6] * tx + inverse_linear[7] * ty + inverse_linear[8] * tz),
    ];
    Ok(row_major_to_columns([
        inverse_linear[0],
        inverse_linear[1],
        inverse_linear[2],
        inverse_translation[0],
        inverse_linear[3],
        inverse_linear[4],
        inverse_linear[5],
        inverse_translation[1],
        inverse_linear[6],
        inverse_linear[7],
        inverse_linear[8],
        inverse_translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]))
}


pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let length = dot3(v, v).sqrt();
    [v[0] / length, v[1] / length, v[2] / length]
}

pub fn coordinate_system3(z: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let sign = if z[2].is_sign_negative() { -1.0 } else { 1.0 };
    let a = -1.0 / (sign + z[2]);
    let b = z[0] * z[1] * a;
    (
        [1.0 + sign * z[0] * z[0] * a, sign * b, -sign * z[0]],
        [b, sign + z[1] * z[1] * a, -z[1]],
    )
}

pub fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}


pub fn row_major_to_columns(matrix: [f32; 16]) -> [[f32; 4]; 4] {
    [
        [matrix[0], matrix[4], matrix[8], matrix[12]],
        [matrix[1], matrix[5], matrix[9], matrix[13]],
        [matrix[2], matrix[6], matrix[10], matrix[14]],
        [matrix[3], matrix[7], matrix[11], matrix[15]],
    ]
}

pub fn inverse_transpose_linear(
    matrix: [f32; 16],
    label: &str,
) -> Result<[[f32; 4]; 4], PbrtError> {
    validate_affine(matrix, label)?;
    let [a, b, c, _, d, e, f, _, g, h, i, _, _, _, _, _] = matrix;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(PbrtError::error(&format!(
            "{label} transform has a singular linear part."
        )));
    }
    let inverse_determinant = 1.0 / determinant;
    Ok(row_major_to_columns([
        (e * i - f * h) * inverse_determinant,
        (f * g - d * i) * inverse_determinant,
        (d * h - e * g) * inverse_determinant,
        0.0,
        (c * h - b * i) * inverse_determinant,
        (a * i - c * g) * inverse_determinant,
        (b * g - a * h) * inverse_determinant,
        0.0,
        (b * f - c * e) * inverse_determinant,
        (c * d - a * f) * inverse_determinant,
        (a * e - b * d) * inverse_determinant,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]))
}

pub fn validate_affine(matrix: [f32; 16], label: &str) -> Result<(), PbrtError> {
    if !matrix.iter().all(|value| value.is_finite()) {
        return Err(PbrtError::error(&format!(
            "{label} transform contains a non-finite value."
        )));
    }
    // Camera transforms are inverted and composed in Float before being
    // narrowed to f32 for the GPU. Keep the affine check strict enough to
    // reject projective transforms while accepting roundoff in the final row.
    let affine_tolerance = 16.0 * f32::EPSILON;
    if matrix[12..15]
        .iter()
        .any(|value| value.abs() > affine_tolerance)
        || (matrix[15] - 1.0).abs() > affine_tolerance
    {
        return Err(PbrtError::error(&format!(
            "{label} transform must be affine; bottom row is {:?}.",
            &matrix[12..16]
        )));
    }
    let determinant = matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
        - matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8])
        + matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8]);
    if determinant == 0.0 {
        return Err(PbrtError::error(&format!(
            "{label} transform is not invertible."
        )));
    }
    Ok(())
}

pub fn row_major_to_tlas_transform(matrix: [f32; 16]) -> [f32; 12] {
    [
        matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5], matrix[6], matrix[7],
        matrix[8], matrix[9], matrix[10], matrix[11],
    ]
}

