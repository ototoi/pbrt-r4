fn write_continuation_dispatch_args(slot: u32, count: u32) {
    let n = (count + INDIRECT_WORKGROUP_SIZE - 1u) / INDIRECT_WORKGROUP_SIZE;
    var x = n;
    var y = 1u;
    if (n > INDIRECT_MAX_WORKGROUPS_X) {
        x = INDIRECT_MAX_WORKGROUPS_X;
        y = (n + INDIRECT_MAX_WORKGROUPS_X - 1u) / INDIRECT_MAX_WORKGROUPS_X;
    }
    queue_dispatch_args[slot] = DispatchIndirectArgs(x, y, 1u);
}

@compute @workgroup_size(1)
fn advance_shadow_continuations() {
    let count = min(
        atomicLoad(&queue_counters.shadow_continuation.count),
        queue_counters.shadow_continuation.capacity,
    );
    atomicStore(&queue_counters.shadow_active.count, count);
    atomicStore(&queue_counters.shadow_continuation.count, 0u);
    write_continuation_dispatch_args(QUEUE_DISPATCH_SLOT_SHADOW_CONTINUATION, count);
}

@compute @workgroup_size(1)
fn advance_medium_continuations() {
    let count = min(
        atomicLoad(&queue_counters.medium_continuation.count),
        queue_counters.medium_continuation.capacity,
    );
    atomicStore(&queue_counters.medium_active.count, count);
    atomicStore(&queue_counters.medium_continuation.count, 0u);
    write_continuation_dispatch_args(QUEUE_DISPATCH_SLOT_MEDIUM_CONTINUATION, count);
}
