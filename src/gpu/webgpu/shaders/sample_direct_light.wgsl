@compute @workgroup_size(64, 1, 1)
fn sample_direct_light(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= direct_eval_count()) { return; }
    let ray_index = load_direct_eval_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let wo = -ray.direction.xyz;
    let material_kind = resolve_attributes_eval_work_item(surface.attributes_eval_work_item).bxdf_kind;
    let offset_direction = select(wo, -wo, material_kind == MATERIAL_KIND_DIFFUSE_TRANSMISSION);
    let light_sample_origin = select(
        offset_ray_origin(
            surface.position.xyz, surface.position_error.xyz,
            surface.geometric_normal.xyz, offset_direction,
        ),
        surface.position.xyz,
        (material_kind == MATERIAL_KIND_DIELECTRIC || material_kind == MATERIAL_KIND_SUBSURFACE)
            && (surface.flags & SURFACE_FLAG_SUBSURFACE_EXIT) == 0u,
    );
    direct_light_samples[pixel_index] = sample_direct_light_at(
        surface.position.xyz,
        surface.normal.xyz,
        light_sample_origin,
        load_sample_lambda(pixel_index),
        load_ray_samples(pixel_index),
    );
}
