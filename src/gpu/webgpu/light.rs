//! Backend-independent light-distribution math used by the WebGPU contract.

/// Compute a triangle's world-space area after applying a row-major transform.
pub fn triangle_world_area(transform: [f32; 16], positions: [[f32; 4]; 3]) -> Option<f32> {
    let [p0, p1, p2] = positions.map(|point| transform_point(transform, point));
    let edge0 = sub(p1, p0);
    let edge1 = sub(p2, p0);
    let cross = cross(edge0, edge1);
    let area = 0.5 * length(cross);
    (area.is_finite() && area > 0.0).then_some(area)
}

/// Return the PMF of one light in the initial uniform light sampler.
pub fn uniform_light_pmf(light_count: u32) -> f32 {
    if light_count == 0 {
        0.0
    } else {
        1.0 / light_count as f32
    }
}

/// Convert an area-measure density into a solid-angle density.
pub fn area_pdf_omega(distance_squared: f32, cosine_at_light: f32, total_area: f32) -> f32 {
    if !distance_squared.is_finite()
        || !cosine_at_light.is_finite()
        || !total_area.is_finite()
        || distance_squared <= 0.0
        || cosine_at_light == 0.0
        || total_area <= 0.0
    {
        0.0
    } else {
        distance_squared / (cosine_at_light.abs() * total_area)
    }
}

fn transform_point(matrix: [f32; 16], point: [f32; 4]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
        matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
        matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}
