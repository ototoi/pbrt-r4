@compute @workgroup_size(8, 8, 1)
fn reset_next_ray_queue(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u && global_id.y == 0u) {
        atomicStore(&queue_counters.next.count, 0u);
        atomicStore(&queue_counters.next.overflow, 0u);
    }
}
