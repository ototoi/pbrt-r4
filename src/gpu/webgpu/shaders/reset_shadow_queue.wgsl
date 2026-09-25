@compute @workgroup_size(1)
fn reset_shadow_queue() {
    atomicStore(&queue_counters.shadow.count, 0u);
    atomicStore(&queue_counters.shadow.overflow, 0u);
}
