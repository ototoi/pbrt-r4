fn transform(matrix: mat4x4<f32>, value: vec4<f32>) -> vec4<f32> {
    return matrix * value;
}
