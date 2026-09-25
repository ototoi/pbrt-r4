fn scattering_local(w: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let t = make_tangent(n);
    return vec3<f32>(dot(w, t), dot(w, cross(n, t)), dot(w, n));
}

fn scattering_local_frame(w: vec3<f32>, tangent: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(w, tangent), dot(w, cross(normal, tangent)), dot(w, normal));
}

fn scattering_world_frame(w: vec3<f32>, tangent: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return tangent * w.x + cross(normal, tangent) * w.y + normal * w.z;
}

fn make_tangent(normal: vec3<f32>) -> vec3<f32> {
    if (abs(normal.x) > 0.1) {
        return normalize(cross(vec3<f32>(0.0, 1.0, 0.0), normal));
    }
    return normalize(cross(vec3<f32>(1.0, 0.0, 0.0), normal));
}

// pbrt-v4 CoordinateSystem(): return the x axis paired with a unit z axis.
fn coordinate_system_x(z: vec3<f32>) -> vec3<f32> {
    let sign = select(1.0, -1.0, (bitcast<u32>(z.z) & 0x80000000u) != 0u);
    let a = -1.0 / (sign + z.z);
    let b = z.x * z.y * a;
    return vec3<f32>(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
}
