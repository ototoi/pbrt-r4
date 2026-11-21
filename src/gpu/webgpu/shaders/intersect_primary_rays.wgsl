@compute @workgroup_size(64, 1, 1)
fn intersect_primary_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= atomicLoad(&queue_counters.medium_active.count)) {
        return;
    }
    let ray = load_current_ray(active_medium_indices[queue_index]);
    trace_closest_surface(ray.origin.xyz, ray.direction.xyz, ray.pixel_index);
}
