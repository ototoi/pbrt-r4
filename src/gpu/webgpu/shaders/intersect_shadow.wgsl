@compute @workgroup_size(8, 8, 1)
fn intersect_shadow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= shadow_ray_count()) {
        return;
    }
    let pixel_index = load_shadow_pixel(ray_index);
    let shadow_origin = load_shadow_vec3(ray_index, 0u);
    let shadow_direction = load_shadow_vec3(ray_index, 3u);
    let shadow_t = load_shadow_t(ray_index);
    if (shadow_t <= 0.0) {
        return;
    }
    let surface = surfaces[pixel_index];
    let shadow_direct = load_shadow_direct(ray_index);
    var query: ray_query;
    rayQueryInitialize(
        &query,
        tlas,
        RayDesc(
            0u,
            0xffu,
            0.0,
            shadow_t,
            shadow_origin,
            shadow_direction,
        ),
    );
    while (rayQueryProceed(&query)) {
    }
    let intersection = rayQueryGetCommittedIntersection(&query);
    if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
        let ray_index = find_current_ray_for_pixel(pixel_index);
        if (ray_index != 0xffffffffu) {
            let ray = load_current_ray(ray_index);
            store_sample_radiance(
                pixel_index,
                load_sample_radiance(pixel_index) + ray.throughput * vec4<f32>(shadow_direct, 0.0),
            );
        }
    } else {
    }
}
