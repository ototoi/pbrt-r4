@compute @workgroup_size(8, 8, 1)
fn reset_classification_queues(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u && global_id.y == 0u) {
        atomicStore(&queue_counters.material.count, 0u);
        atomicStore(&queue_counters.material.overflow, 0u);
        atomicStore(&queue_counters.hit_area.count, 0u);
        atomicStore(&queue_counters.hit_area.overflow, 0u);
        atomicStore(&queue_counters.escaped.count, 0u);
        atomicStore(&queue_counters.escaped.overflow, 0u);
    }
}
