fn gamma(n: f32) -> f32 {
    return (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON);
}

// pbrt-v4 bounds only the barycentric interpolation error and relies on its
// watertight intersector rejecting t == 0 for rays spawned on a surface.
// Hardware ray queries give no such guarantee: a point on an axis-aligned
// plane through the origin has zero error along the normal, so the spawned
// ray starts exactly on the plane and can re-hit its own triangle. Pad the
// bound by the triangle's coordinate magnitude, which is what the hardware
// intersector's error scales with (not the hit point, which may be ~0).
const HARDWARE_INTERSECTION_ERROR_GAMMA_N: f32 = 8.0;

fn hardware_intersection_error(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec3<f32> {
    let m = max(max(abs(p0), abs(p1)), abs(p2));
    let magnitude = max(max(m.x, m.y), m.z);
    return vec3<f32>(gamma(HARDWARE_INTERSECTION_ERROR_GAMMA_N) * magnitude);
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
