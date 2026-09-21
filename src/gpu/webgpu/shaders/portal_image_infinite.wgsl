struct PortalUvResult { uv: vec2<f32>, duv_dw: f32, valid: u32 };
struct PortalDirectionResult { wi: vec3<f32>, duv_dw: f32, valid: u32 };
struct PortalBoundsResult { min: vec2<f32>, max: vec2<f32>, valid: u32 };
struct PortalSampleResult { uv: vec2<f32>, pdf: f32, valid: u32 };
struct PortalAxisSampleResult { value: f32, valid: u32 };

fn portal_finite(value: f32) -> bool {
    return value == value && abs(value) <= RAY_T_MAX;
}

fn portal_finite2(value: vec2<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec2<f32>(RAY_T_MAX));
}

fn portal_finite3(value: vec3<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec3<f32>(RAY_T_MAX));
}

fn portal_image_from_render(portal: PortalImageInfiniteRecord, w_render: vec3<f32>) -> PortalUvResult {
    let w = normalize(vec3<f32>(
        dot(portal.world_to_portal0.xyz, w_render),
        dot(portal.world_to_portal1.xyz, w_render),
        dot(portal.world_to_portal2.xyz, w_render),
    ));
    if (!portal_finite3(w) || w.z <= 0.0) {
        return PortalUvResult(vec2<f32>(0.0), 0.0, 0u);
    }
    let duv_dw = PI * PI * (1.0 - w.x * w.x) * (1.0 - w.y * w.y) / w.z;
    let uv = clamp(
        vec2<f32>(atan2(w.x, w.z), atan2(w.y, w.z)) / PI + vec2<f32>(0.5),
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    if (!portal_finite(duv_dw) || !portal_finite2(uv)) {
        return PortalUvResult(vec2<f32>(0.0), 0.0, 0u);
    }
    return PortalUvResult(uv, duv_dw, 1u);
}

fn portal_render_from_image(portal: PortalImageInfiniteRecord, uv: vec2<f32>) -> PortalDirectionResult {
    let angles = -vec2<f32>(0.5 * PI) + uv * PI;
    let w = normalize(vec3<f32>(tan(angles.x), tan(angles.y), 1.0));
    if (!portal_finite3(w) || w.z <= 0.0) {
        return PortalDirectionResult(vec3<f32>(0.0), 0.0, 0u);
    }
    let duv_dw = PI * PI * (1.0 - w.x * w.x) * (1.0 - w.y * w.y) / w.z;
    let wi = normalize(
        portal.world_to_portal0.xyz * w.x
            + portal.world_to_portal1.xyz * w.y
            + portal.world_to_portal2.xyz * w.z,
    );
    if (!portal_finite(duv_dw) || !portal_finite3(wi)) {
        return PortalDirectionResult(vec3<f32>(0.0), 0.0, 0u);
    }
    return PortalDirectionResult(wi, duv_dw, 1u);
}

fn portal_image_bounds(portal: PortalImageInfiniteRecord, p: vec3<f32>) -> PortalBoundsResult {
    let p0 = portal_image_from_render(portal, normalize(portal.portal0.xyz - p));
    let p1 = portal_image_from_render(portal, normalize(portal.portal2.xyz - p));
    if (p0.valid == 0u || p1.valid == 0u) {
        return PortalBoundsResult(vec2<f32>(0.0), vec2<f32>(0.0), 0u);
    }
    return PortalBoundsResult(min(p0.uv, p1.uv), max(p0.uv, p1.uv), 1u);
}

fn portal_sat_lookup_int(portal: PortalImageInfiniteRecord, x_in: i32, y_in: i32) -> f32 {
    if (x_in == 0 || y_in == 0) {
        return 0.0;
    }
    let x = min(x_in - 1, i32(portal.width) - 1);
    let y = min(y_in - 1, i32(portal.height) - 1);
    let index = portal.distribution_offset + u32(y) * portal.width + u32(x);
    return portal_distribution[index].summed_area;
}

fn portal_sat_lookup(portal: PortalImageInfiniteRecord, p: vec2<f32>) -> f32 {
    let scaled = p * vec2<f32>(f32(portal.width), f32(portal.height));
    let p0 = vec2<i32>(scaled);
    let d = scaled - vec2<f32>(p0);
    let v00 = portal_sat_lookup_int(portal, p0.x, p0.y);
    let v10 = portal_sat_lookup_int(portal, p0.x + 1, p0.y);
    let v01 = portal_sat_lookup_int(portal, p0.x, p0.y + 1);
    let v11 = portal_sat_lookup_int(portal, p0.x + 1, p0.y + 1);
    return (1.0 - d.x) * (1.0 - d.y) * v00
        + d.x * (1.0 - d.y) * v10
        + (1.0 - d.x) * d.y * v01
        + d.x * d.y * v11;
}

fn portal_sat_integral(portal: PortalImageInfiniteRecord, lower: vec2<f32>, upper: vec2<f32>) -> f32 {
    let sum = portal_sat_lookup(portal, upper)
        - portal_sat_lookup(portal, vec2<f32>(lower.x, upper.y))
        + portal_sat_lookup(portal, lower)
        - portal_sat_lookup(portal, vec2<f32>(upper.x, lower.y));
    return max(sum / f32(portal.width * portal.height), 0.0);
}

fn portal_distribution_eval(portal: PortalImageInfiniteRecord, uv: vec2<f32>) -> f32 {
    let x = min(u32(uv.x * f32(portal.width)), portal.width - 1u);
    let y = min(u32(uv.y * f32(portal.height)), portal.height - 1u);
    return portal_distribution[portal.distribution_offset + y * portal.width + x].function;
}

fn portal_sample_marginal_x(
    portal: PortalImageInfiniteRecord,
    u: f32,
    bounds: PortalBoundsResult,
    bounds_integral: f32,
) -> PortalAxisSampleResult {
    var x_min = bounds.min.x;
    var x_max = bounds.max.x;
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (ceil(f32(portal.width) * x_max) - floor(f32(portal.width) * x_min) <= 1.0) {
            break;
        }
        let mid = 0.5 * (x_min + x_max);
        let cdf = portal_sat_integral(portal, bounds.min, vec2<f32>(mid, bounds.max.y)) / bounds_integral;
        if (cdf > u) { x_max = mid; } else { x_min = mid; }
    }
    let px_min = portal_sat_integral(portal, bounds.min, vec2<f32>(x_min, bounds.max.y)) / bounds_integral;
    let px_max = portal_sat_integral(portal, bounds.min, vec2<f32>(x_max, bounds.max.y)) / bounds_integral;
    let cdf_delta = px_max - px_min;
    if (!portal_finite(px_min) || !portal_finite(px_max)
        || !portal_finite(cdf_delta) || cdf_delta == 0.0) {
        return PortalAxisSampleResult(0.0, 0u);
    }
    let x = clamp(mix(x_min, x_max, (u - px_min) / cdf_delta), x_min, x_max);
    if (!portal_finite(x)) { return PortalAxisSampleResult(0.0, 0u); }
    return PortalAxisSampleResult(x, 1u);
}

fn portal_sample_conditional_y(
    portal: PortalImageInfiniteRecord,
    u: f32,
    bounds: PortalBoundsResult,
    x: f32,
) -> PortalAxisSampleResult {
    let nx = f32(portal.width);
    let cond_min = vec2<f32>(floor(x * nx) / nx, bounds.min.y);
    var cond_max = vec2<f32>(ceil(x * nx) / nx, bounds.max.y);
    if (cond_min.x == cond_max.x) { cond_max.x += 1.0 / nx; }
    let cond_integral = portal_sat_integral(portal, cond_min, cond_max);
    if (!portal_finite(cond_integral) || cond_integral == 0.0) {
        return PortalAxisSampleResult(0.0, 0u);
    }
    var y_min = bounds.min.y;
    var y_max = bounds.max.y;
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (ceil(f32(portal.height) * y_max) - floor(f32(portal.height) * y_min) <= 1.0) {
            break;
        }
        let mid = 0.5 * (y_min + y_max);
        let cdf = portal_sat_integral(portal, cond_min, vec2<f32>(cond_max.x, mid)) / cond_integral;
        if (cdf > u) { y_max = mid; } else { y_min = mid; }
    }
    let py_min = portal_sat_integral(portal, cond_min, vec2<f32>(cond_max.x, y_min)) / cond_integral;
    let py_max = portal_sat_integral(portal, cond_min, vec2<f32>(cond_max.x, y_max)) / cond_integral;
    let cdf_delta = py_max - py_min;
    if (!portal_finite(py_min) || !portal_finite(py_max)
        || !portal_finite(cdf_delta) || cdf_delta == 0.0) {
        return PortalAxisSampleResult(0.0, 0u);
    }
    let y = clamp(mix(y_min, y_max, (u - py_min) / cdf_delta), y_min, y_max);
    if (!portal_finite(y)) { return PortalAxisSampleResult(0.0, 0u); }
    return PortalAxisSampleResult(y, 1u);
}

fn sample_portal_distribution(portal: PortalImageInfiniteRecord, u: vec2<f32>, bounds: PortalBoundsResult) -> PortalSampleResult {
    if (bounds.valid == 0u || portal.width == 0u || portal.height == 0u
        || !portal_finite2(u)) {
        return PortalSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }
    let bounds_integral = portal_sat_integral(portal, bounds.min, bounds.max);
    if (!portal_finite(bounds_integral) || bounds_integral == 0.0) {
        return PortalSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }
    let x = portal_sample_marginal_x(portal, u.x, bounds, bounds_integral);
    if (x.valid == 0u) {
        return PortalSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }
    let y = portal_sample_conditional_y(portal, u.y, bounds, x.value);
    if (y.valid == 0u) {
        return PortalSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }
    let uv = vec2<f32>(x.value, y.value);
    let pdf = portal_distribution_eval(portal, uv) / bounds_integral;
    if (!portal_finite(pdf)) {
        return PortalSampleResult(vec2<f32>(0.0), 0.0, 0u);
    }
    return PortalSampleResult(uv, pdf, 1u);
}

fn portal_distribution_pdf(portal: PortalImageInfiniteRecord, uv: vec2<f32>, bounds: PortalBoundsResult) -> f32 {
    if (bounds.valid == 0u || portal.width == 0u || portal.height == 0u) {
        return 0.0;
    }
    let integral = portal_sat_integral(portal, bounds.min, bounds.max);
    if (!portal_finite(integral) || integral == 0.0) { return 0.0; }
    let pdf = portal_distribution_eval(portal, uv) / integral;
    if (!portal_finite(pdf)) { return 0.0; }
    return pdf;
}

fn load_portal_image_spectrum(light_index: u32, uv: vec2<f32>, lambda: vec4<f32>) -> vec4<f32> {
    let record = light_records[light_index];
    let payload = record.sampling_model;
    let model = light_sampling_models[payload];
    let binding = model.flags & 0x0fffffffu;
    let rgb = max(textureSampleLevel(texture_images[binding], texture_samplers[0], uv, 0.0).rgb, vec3<f32>(0.0));
    let illuminant = load_light_spectrum(light_index, 2u, lambda);
    return rgb_to_unbounded_spectrum4(rgb, lambda, model.flags >> 28u) * illuminant;
}
