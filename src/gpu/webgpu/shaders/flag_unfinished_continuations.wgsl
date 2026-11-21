@compute @workgroup_size(1)
fn flag_unfinished_continuations() {
    if (atomicLoad(&queue_counters.medium_continuation.count) != 0u
        || atomicLoad(&queue_counters.shadow_continuation.count) != 0u) {
        sample_retry.pending = 1u;
    }
    atomicStore(&queue_counters.medium_continuation.count, 0u);
    atomicStore(&queue_counters.shadow_continuation.count, 0u);
}
