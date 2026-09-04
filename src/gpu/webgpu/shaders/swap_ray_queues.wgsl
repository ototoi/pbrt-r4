@compute @workgroup_size(8, 8, 1)
fn swap_ray_queues(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * viewport.width + global_id.x;
    let next_count = atomicLoad(&next_ray_queue_state.count);
    if (index == 0u) {
        atomicStore(&current_ray_queue_state.count, next_count);
    }
    if (index >= next_count || index >= current_ray_queue_state.capacity) {
        return;
    }
    current_ray_queue[index] = next_ray_queue[index];
}
