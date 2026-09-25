// Copies the surviving rays queued for the next depth into the current-ray
// queue. reset_next_ray_queue commits the new current-ray count separately,
// since it must run even when there is nothing to copy.
@compute @workgroup_size(64, 1, 1)
fn swap_ray_queues(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (index >= next_ray_count()) {
        return;
    }
    store_current_ray(index, load_next_ray(index));
}
