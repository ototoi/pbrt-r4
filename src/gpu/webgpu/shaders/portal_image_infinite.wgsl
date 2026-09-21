struct PortalUvResult { uv: vec2<f32>, duv_dw: f32, valid: u32 };
struct PortalDirectionResult { wi: vec3<f32>, duv_dw: f32, valid: u32 };
struct PortalBoundsResult { min: vec2<f32>, max: vec2<f32>, valid: u32 };
struct PortalSampleResult { uv: vec2<f32>, pdf: f32, valid: u32 };

fn portal_equal_area_square_to_sphere(p: vec2<f32>) -> vec3<f32> {
    let q = 2.0 * p - vec2<f32>(1.0);
    let z = 1.0 - abs(q.x) - abs(q.y);
    let d = max(1.0 - abs(z), 1e-6);
    return normalize(vec3<f32>(q.x * d, q.y * d, z));
}

fn portal_image_from_render(portal: PortalImageInfiniteRecord, w: vec3<f32>) -> PortalUvResult {
    let local = vec3<f32>(dot(portal.world_to_portal0.xyz, w), dot(portal.world_to_portal1.xyz, w), dot(portal.world_to_portal2.xyz, w));
    if (!all(local == local) || local.z <= 0.0) { return PortalUvResult(vec2<f32>(0.0), 0.0, 0u); }
    let uv = equal_area_sphere_to_square(normalize(local));
    return PortalUvResult(uv, 1.0, 1u);
}

fn portal_render_from_image(portal: PortalImageInfiniteRecord, uv: vec2<f32>) -> PortalDirectionResult {
    let local = portal_equal_area_square_to_sphere(uv);
    let wi = portal.world_to_portal0.xyz * local.x + portal.world_to_portal1.xyz * local.y + portal.world_to_portal2.xyz * local.z;
    if (!all(wi == wi)) { return PortalDirectionResult(vec3<f32>(0.0), 0.0, 0u); }
    return PortalDirectionResult(wi, 1.0, 1u);
}

fn portal_image_bounds(portal: PortalImageInfiniteRecord, p: vec3<f32>) -> PortalBoundsResult {
    let corners = array<vec3<f32>, 4>(portal.portal0.xyz, portal.portal1.xyz, portal.portal2.xyz, portal.portal3.xyz);
    var lower = vec2<f32>(1.0);
    var upper = vec2<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let d = normalize(corners[i] - p);
        let mapped = portal_image_from_render(portal, d);
        if (mapped.valid == 0u) { return PortalBoundsResult(vec2<f32>(0.0), vec2<f32>(0.0), 0u); }
        lower = min(lower, mapped.uv);
        upper = max(upper, mapped.uv);
    }
    return PortalBoundsResult(lower, upper, select(0u, 1u, all(lower <= upper)));
}

fn sample_portal_distribution(portal: PortalImageInfiniteRecord, u: vec2<f32>, bounds: PortalBoundsResult) -> PortalSampleResult {
    if (bounds.valid == 0u || portal.width == 0u || portal.height == 0u) { return PortalSampleResult(vec2<f32>(0.0), 0.0, 0u); }
    let uv = bounds.min + u * (bounds.max - bounds.min);
    return PortalSampleResult(uv, 1.0, 1u);
}

fn portal_distribution_pdf(portal: PortalImageInfiniteRecord, uv: vec2<f32>, bounds: PortalBoundsResult) -> f32 {
    if (bounds.valid == 0u || any(uv < bounds.min) || any(uv > bounds.max)) { return 0.0; }
    return 1.0;
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
