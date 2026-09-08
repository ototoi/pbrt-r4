@compute @workgroup_size(8, 8, 1)
fn sample_composite_bounce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) { return; }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= current_ray_count()) { return; }
    let ray = load_current_ray(ray_index);
    let surface = surfaces[ray.pixel_index];
    if (surface.hit == 0u || surface.flags != 0u) { return; }
    let kind = load_material_kind(surface.material);
    if (kind != MATERIAL_KIND_COATED_DIFFUSE && kind != MATERIAL_KIND_COATED_CONDUCTOR) { return; }
    let root = load_evaluated_attributes(surface.evaluated_attributes_root);
    let coat = load_evaluated_attributes(root.child_work_item0);
    let normal = normalize(surface.normal.xyz);
    let wo = normalize(-ray.direction.xyz);
    let direction = normalize(reflect(-wo, normal));
    let cosine = abs(dot(normal, wo));
    if (cosine <= 1e-5) { return; }
    let eta = max(coat.values[0].x, 1.0001);
    let fresnel = dielectric_fresnel(cosine, eta);
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz, surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0),
        ray.throughput * fresnel,
        surface.position, surface.position_error, surface.geometric_normal,
        vec4<f32>(normal, 0.0), ray.pixel_index, ray.depth + 1u,
        ray.inv_w_u, ray.inv_w_u / max(fresnel, 1e-7), fresnel,
        0u, 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= queue_counters.next.capacity) {
        atomicStore(&queue_counters.next.overflow, 1u);
        return;
    }
    store_next_ray(next_index, next_ray);
}
