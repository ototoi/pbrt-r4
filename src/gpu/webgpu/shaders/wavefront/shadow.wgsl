@compute @workgroup_size(64)
fn intersect_shadow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot_index = global_id.x;
    if (slot_index >= pixel_count() || slot_index >= arena.capacity) {
        return;
    }

    let ray = arena.rays[slot_index];
    if (ray.indices.y != RAY_STATE_SHADOW) {
        return;
    }

    if (ray.hit.x <= 0.0) {
        arena.rays[slot_index].direct_lighting.w = 0.0;
        arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
        return;
    }
    var query: ray_query;
    rayQueryInitialize(
        &query,
        acceleration,
        RayDesc(0u, 0xFFu, 0.0, ray.hit.x, ray.origin.xyz, ray.direction.xyz),
    );
    rayQueryProceed(&query);
    let intersection = rayQueryGetCommittedIntersection(&query);
    if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
        arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
    } else {
        arena.rays[slot_index].direct_lighting.w = 0.0;
        arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
    }
}
