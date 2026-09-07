@compute @workgroup_size(8, 8, 1)
fn sample_thin_dielectric_bounce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) { return; }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= current_ray_count()) { return; }
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    if (surface.hit == 0u || surface.flags != 0u
        || load_material_kind(surface.material) != MATERIAL_KIND_THIN_DIELECTRIC) { return; }
    let eta_node = load_material_surface_node(surface.material);
    if (!dielectric_eta_is_constant(eta_node)) { terminate_secondary_wavelengths(pixel_index); }
    var eta = load_dielectric_eta(eta_node, load_sample_lambda(pixel_index)).x;
    if (eta == 0.0) { eta = 1.0; }
    let wo = normalize(-ray.direction.xyz);
    let r0 = layered_fresnel(dot(wo, normalize(surface.normal.xyz)), eta);
    let r = select(r0, r0 + (1.0 - r0) * (1.0 - r0) * r0 / max(1.0 - r0 * r0, 1e-7), r0 < 1.0);
    let t = 1.0 - r;
    if (!(eta > 0.0) || eta != eta || !(r >= 0.0) || !(r <= 1.0)) {
        set_render_error(); return;
    }
    let samples = load_ray_samples(pixel_index);
    let reflection = samples.indirect.x < r;
    let direction = select(-wo, vec3<f32>(-wo.x, -wo.y, wo.z), reflection);
    let probability = max(select(t, r, reflection), 1e-7);
    let f = select(vec4<f32>(t / abs(direction.z)), vec4<f32>(r / abs(direction.z)), reflection);
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz,
            surface.geometric_normal.xyz, direction), 1.0), vec4<f32>(direction, 0.0),
        ray.throughput * f / probability, surface.position,
        surface.position_error, surface.geometric_normal, surface.normal,
        pixel_index, ray.depth + 1u, ray.inv_w_u, ray.inv_w_u / probability,
        probability, 0u, 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) { atomicStore(&queue_counters.next.overflow, 1u); return; }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
