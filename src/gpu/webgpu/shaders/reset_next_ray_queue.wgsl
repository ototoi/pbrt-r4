@compute @workgroup_size(8, 8, 1)
fn reset_next_ray_queue(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u && global_id.y == 0u) {
        atomicStore(&wavefront_queue[NEXT_COUNT], 0u);
        atomicStore(&wavefront_queue[NEXT_OVERFLOW], 0u);
    }
}
