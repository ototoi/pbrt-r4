@compute @workgroup_size(8, 8, 1)
fn prepare_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let pixel_index = global_id.y * viewport.width + global_id.x;
    if (pixel_index == 0u) {
        atomicStore(&queue_counters.current.count, 0u);
        atomicStore(&queue_counters.current.overflow, 0u);
        atomicStore(&queue_counters.next.count, 0u);
        atomicStore(&queue_counters.next.overflow, 0u);
        atomicStore(&queue_counters.shadow.count, 0u);
        atomicStore(&queue_counters.shadow.overflow, 0u);
        atomicStore(&queue_counters.material.count, 0u);
        atomicStore(&queue_counters.material.overflow, 0u);
        atomicStore(&queue_counters.hit_area.count, 0u);
        atomicStore(&queue_counters.hit_area.overflow, 0u);
        atomicStore(&queue_counters.escaped.count, 0u);
        atomicStore(&queue_counters.escaped.overflow, 0u);
        if (viewport.sample_index == 0u) {
            atomicStore(&render_error.value, 0u);
        }
    }
    store_sample_radiance(pixel_index, vec4<f32>(0.0));
    store_ray_samples(pixel_index, RaySamples(vec4<f32>(0.0), vec4<f32>(0.0)));
    surfaces[pixel_index].hit = 0u;
    surfaces[pixel_index].flags = 0u;
}
