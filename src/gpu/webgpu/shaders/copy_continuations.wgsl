// Kept apart from advance_continuations.wgsl: binding the dispatch arguments
// here would conflict with their use as this stage's indirect buffer.
@compute @workgroup_size(64, 1, 1)
fn copy_shadow_continuations(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (index >= atomicLoad(&queue_counters.shadow_active.count)) { return; }
    active_shadow_indices[index] = next_shadow_indices[index];
}

@compute @workgroup_size(64, 1, 1)
fn copy_medium_continuations(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (index >= atomicLoad(&queue_counters.medium_active.count)) { return; }
    active_medium_indices[index] = next_medium_indices[index];
}
