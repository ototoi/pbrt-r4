//! Backend-independent light-distribution math used by the WebGPU contract.

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
