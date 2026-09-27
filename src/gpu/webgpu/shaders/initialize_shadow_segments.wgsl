@compute @workgroup_size(64, 1, 1)
fn initialize_shadow_segments(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    let count = shadow_ray_count();
    if (index >= count) { return; }
    active_shadow_indices[index] = index;
    if (index == 0u) {
        atomicStore(&queue_counters.shadow_active.count, count);
    }
}
