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
