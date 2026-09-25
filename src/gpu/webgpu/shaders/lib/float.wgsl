fn gamma(n: f32) -> f32 {
    return (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON);
}

fn next_float_up(value: f32) -> f32 {
    if (value == 0.0) {
        return bitcast<f32>(1u);
    }
    let bits = bitcast<u32>(value);
    if (value < 0.0) {
        return bitcast<f32>(bits - 1u);
    }
    return bitcast<f32>(bits + 1u);
}

fn next_float_down(value: f32) -> f32 {
    if (value == 0.0) {
        return bitcast<f32>(0x80000001u);
    }
    let bits = bitcast<u32>(value);
    if (value > 0.0) {
        return bitcast<f32>(bits - 1u);
    }
    return bitcast<f32>(bits + 1u);
}
