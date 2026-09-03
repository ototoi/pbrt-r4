@compute @workgroup_size(8, 8, 1)
fn intersect_shadow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let pixel_index = global_id.y * viewport.width + global_id.x;
    let ray_index = current_ray_index(pixel_index);
    if (rays[ray_index].is_active == 0u || surfaces[pixel_index].shadow_t <= 0.0) {
        return;
    }
    let surface = surfaces[pixel_index];
    var query: ray_query;
    rayQueryInitialize(
        &query,
        tlas,
        RayDesc(
            0u,
            0xffu,
            0.0,
            surface.shadow_t,
            surface.shadow_origin.xyz,
            surface.shadow_direction.xyz,
        ),
    );
    while (rayQueryProceed(&query)) {
    }
    let intersection = rayQueryGetCommittedIntersection(&query);
    if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
        surfaces[pixel_index].shadow_visible = 1u;
    } else {
        surfaces[pixel_index].shadow_visible = 2u;
    }
}
