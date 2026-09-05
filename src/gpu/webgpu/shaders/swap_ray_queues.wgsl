@compute @workgroup_size(8, 8, 1)
fn swap_ray_queues(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * viewport.width + global_id.x;
    let next_count = next_ray_count();
    if (index == 0u) {
        atomicStore(&wavefront_queue[CURRENT_COUNT], next_count);
    }
    if (index >= next_count || index >= pixel_count()) {
        return;
    }
    store_current_ray(index, load_next_ray(index));
}
