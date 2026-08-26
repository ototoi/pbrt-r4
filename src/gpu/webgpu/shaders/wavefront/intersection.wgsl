@group(0) @binding(2)
var acceleration: acceleration_structure;

@compute @workgroup_size(64)
fn intersect_closest(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot_index = global_id.x;
    if (slot_index >= pixel_count() || slot_index >= arena.capacity) {
        return;
    }

    let ray = arena.rays[slot_index];
    if (ray.indices.y != RAY_STATE_ACTIVE) {
        return;
    }

    var query: ray_query;
    rayQueryInitialize(
        &query,
        acceleration,
        RayDesc(0u, 0xFFu, 0.0, 1.0e30, ray.origin.xyz, ray.direction.xyz),
    );
    rayQueryProceed(&query);
    let intersection = rayQueryGetCommittedIntersection(&query);
    if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
        arena.rays[slot_index].indices.y = RAY_STATE_MISS;
        return;
    }

    arena.rays[slot_index].hit = vec4<f32>(
        intersection.t,
        intersection.barycentrics.x,
        intersection.barycentrics.y,
        ray.hit.w,
    );
    arena.rays[slot_index].indices.y = RAY_STATE_HIT;
    arena.rays[slot_index].indices.z = intersection.instance_custom_data;
    arena.rays[slot_index].indices.w = intersection.primitive_index;
}
