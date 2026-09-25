// Commits the ray queue swap's result (swap_ray_queues has already copied
// the surviving rays into the current-ray queue) and clears the next-ray
// queue for this depth's bounces. Always runs, even when there is nothing to
// swap, since queue_counters.current.count must reach 0 for a depth with no
// surviving rays.
@compute @workgroup_size(1)
fn reset_next_ray_queue() {
    let next_count = next_ray_count();
    atomicStore(&queue_counters.current.count, next_count);
    atomicStore(&queue_counters.next.count, 0u);
    atomicStore(&queue_counters.next.overflow, 0u);
}
