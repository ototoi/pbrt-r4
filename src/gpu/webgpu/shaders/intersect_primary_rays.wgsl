@compute @workgroup_size(8, 8, 1)
fn intersect_primary_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= current_ray_count()) {
        return;
    }
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    if (ray.is_active == 0u) {
        surfaces[pixel_index].hit = 0u;
        return;
    }
    var query: ray_query;
    rayQueryInitialize(
        &query,
        tlas,
        RayDesc(0u, 0xffu, 0.0, RAY_T_MAX, ray.origin.xyz, ray.direction.xyz),
    );
    while (rayQueryProceed(&query)) {
    }
    let intersection = rayQueryGetCommittedIntersection(&query);
    if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
        surfaces[pixel_index].hit = 0u;
    } else {
        surfaces[pixel_index].t = intersection.t;
        surfaces[pixel_index].hit = 1u;
        surfaces[pixel_index].instance_custom_data = intersection.instance_custom_data;
        surfaces[pixel_index].primitive_index = intersection.primitive_index;
        surfaces[pixel_index].barycentric = vec4<f32>(intersection.barycentrics, 0.0, 0.0);
    }
}
