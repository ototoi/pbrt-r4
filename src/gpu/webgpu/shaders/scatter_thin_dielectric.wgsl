@compute @workgroup_size(64, 1, 1)
fn scatter_thin_dielectric(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= scatter_thin_dielectric_count()) { return; }
    let ray_index = load_scatter_thin_dielectric_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);
    let eta_attribute = load_material_attribute(evaluated.material_node, 0u);
    if (eta_attribute.kind != 1u) { terminate_secondary_wavelengths(pixel_index); }
    var eta = evaluated.values[0].x;
    if (eta == 0.0) { eta = 1.0; }
    let wo = normalize(-ray.direction.xyz);
    let r0 = dielectric_fresnel(dot(wo, normalize(surface.normal.xyz)), eta);
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
        probability, 1u, 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) { atomicStore(&queue_counters.next.overflow, 1u); return; }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
