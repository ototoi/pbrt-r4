@compute @workgroup_size(8, 8, 1)
fn sample_portal_direct(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let pixel_index = global_id.y * viewport.width + global_id.x;
    let candidate = portal_light_candidates[pixel_index];
    if (candidate.state != PORTAL_CANDIDATE_SELECTED) { return; }
    portal_light_candidates[pixel_index] = PortalLightCandidate(
        vec4<f32>(0.0), vec4<f32>(0.0), PORTAL_CANDIDATE_INVALID, 0u, 0u, 0u,
    );
    let light_index = candidate.light_index;
    let model = light_sampling_models[load_light_payload(light_index)];
    if (model.geometry_kind != 3u || model.geometry_index >= arrayLength(&portal_infinite_lights)) {
        set_render_error();
        return;
    }
    let portal = portal_infinite_lights[model.geometry_index];
    if (portal.width == 0u || portal.height == 0u) {
        set_render_error();
        return;
    }
    let bounds = portal_image_bounds(portal, candidate.position_uv.xyz);
    let sample = sample_portal_distribution(
        portal, vec2<f32>(candidate.position_uv.w, candidate.sample_direction_pdf.x), bounds,
    );
    if (sample.valid == 0u) { return; }
    let direction = portal_render_from_image(portal, sample.uv);
    if (direction.valid == 0u || direction.duv_dw <= 0.0) { return; }
    let pdf = candidate.sample_direction_pdf.w * sample.pdf / direction.duv_dw;
    if (!portal_finite(pdf) || pdf == 0.0) { return; }
    portal_light_candidates[pixel_index] = PortalLightCandidate(
        vec4<f32>(sample.uv, 0.0, 0.0), vec4<f32>(direction.wi, pdf), PORTAL_CANDIDATE_SAMPLED,
        light_index, 0u, 0u,
    );
}
