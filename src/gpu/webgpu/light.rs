//! Backend-independent light-distribution math used by the WebGPU contract.

use crate::util::error::PbrtError;

/// Build an area-proportional triangle CDF, excluding degenerate triangles.
pub fn build_triangle_cdf(areas: &[(u32, f32)]) -> Result<(f32, Vec<(u32, f32)>), PbrtError> {
    let total_area: f32 = areas
        .iter()
        .filter(|(_, area)| area.is_finite() && *area > 0.0)
        .map(|(_, area)| *area)
        .sum();
    if areas.is_empty() || !total_area.is_finite() || total_area <= 0.0 {
        return Err(PbrtError::error(
            "Area-light triangle distribution is empty.",
        ));
    }
    let valid: Vec<_> = areas
        .iter()
        .copied()
        .filter(|(_, area)| area.is_finite() && *area > 0.0)
        .collect();
    let mut cumulative = 0.0;
    let mut previous = 0.0;
    let mut cdf = Vec::with_capacity(valid.len());
    for (index, (primitive, area)) in valid.iter().enumerate() {
        cumulative += *area / total_area;
        let value = if index + 1 == valid.len() {
            1.0
        } else {
            cumulative
        };
        if !value.is_finite() || (index > 0 && value <= previous) {
            return Err(PbrtError::error(
                "Area-light triangle CDF is not increasing.",
            ));
        }
        cdf.push((*primitive, value));
        previous = value;
    }
    Ok((total_area, cdf))
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
