@compute @workgroup_size(8, 8, 1)
fn select_portal_direct(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) { return; }
    let queue_index = global_id.y * viewport.width + global_id.x;
    if (queue_index >= material_eval_count()) { return; }
    let ray_index = load_material_eval_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    portal_light_candidates[pixel_index] = PortalLightCandidate(
        vec4<f32>(0.0), vec4<f32>(0.0), PORTAL_CANDIDATE_INVALID, 0u, 0u, 0u,
    );
    let surface = surfaces[pixel_index];
    if (surface.hit == 0u || ray.depth >= viewport.max_depth || light_table.light_count == 0u) {
        return;
    }
    let samples = load_ray_samples(pixel_index);
    let selection = sample_scene_light(samples.direct.x, surface.position.xyz, surface.normal.xyz);
    if (selection.pmf <= 0.0 || selection.index == 0xffffffffu
        || load_light_kind(selection.index) != LIGHT_KIND_PORTAL_IMAGE_INFINITE) {
        return;
    }
    portal_light_candidates[pixel_index] = PortalLightCandidate(
        vec4<f32>(surface.position.xyz, samples.direct.y),
        vec4<f32>(samples.direct.z, 0.0, 0.0, selection.pmf), PORTAL_CANDIDATE_SELECTED,
        selection.index, 0u, 0u,
    );
}
