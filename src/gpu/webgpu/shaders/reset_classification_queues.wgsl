@compute @workgroup_size(8, 8, 1)
fn reset_classification_queues(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u && global_id.y == 0u) {
        atomicStore(&wavefront_queue[MATERIAL_COUNT], 0u);
        atomicStore(&wavefront_queue[MATERIAL_OVERFLOW], 0u);
        atomicStore(&wavefront_queue[HIT_AREA_COUNT], 0u);
        atomicStore(&wavefront_queue[HIT_AREA_OVERFLOW], 0u);
    }
}
