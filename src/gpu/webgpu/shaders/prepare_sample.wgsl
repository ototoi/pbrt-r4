@compute @workgroup_size(8, 8, 1)
fn prepare_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let pixel_index = global_id.y * viewport.width + global_id.x;
    if (pixel_index == 0u) {
        atomicStore(&wavefront_queue[CURRENT_COUNT], 0u);
        atomicStore(&wavefront_queue[CURRENT_OVERFLOW], 0u);
        atomicStore(&wavefront_queue[NEXT_COUNT], 0u);
        atomicStore(&wavefront_queue[NEXT_OVERFLOW], 0u);
        atomicStore(&wavefront_queue[SHADOW_COUNT], 0u);
        atomicStore(&wavefront_queue[SHADOW_OVERFLOW], 0u);
        atomicStore(&wavefront_queue[MATERIAL_COUNT], 0u);
        atomicStore(&wavefront_queue[MATERIAL_OVERFLOW], 0u);
        atomicStore(&wavefront_queue[HIT_AREA_COUNT], 0u);
        atomicStore(&wavefront_queue[HIT_AREA_OVERFLOW], 0u);
        atomicStore(&wavefront_queue[ESCAPED_COUNT], 0u);
        atomicStore(&wavefront_queue[ESCAPED_OVERFLOW], 0u);
    }
    store_sample_radiance(pixel_index, vec4<f32>(0.0));
    store_sample_metadata(pixel_index);
    surfaces[pixel_index].hit = 0u;
    surfaces[pixel_index].shadow_visible = 0u;
    surfaces[pixel_index].flags = 0u;
    surfaces[pixel_index].direct = vec4<f32>(0.0);
}
