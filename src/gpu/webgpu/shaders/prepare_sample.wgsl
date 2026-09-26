@compute @workgroup_size(8, 8, 1)
fn prepare_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.tile_width || global_id.y >= viewport.tile_height) {
        return;
    }
    let pixel_index = global_id.y * viewport.tile_width + global_id.x;
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
        // Only the render's very first tile-sample clears the error flag:
        // every tile revisits sample_index == 0u, and a later tile must not
        // erase an error an earlier tile already reported before it's read.
        if (viewport.sample_index == 0u
            && viewport.tile_x == viewport.region_x
            && viewport.tile_y == viewport.region_y) {
            atomicStore(&render_error.value, 0u);
        }
    }
    store_sample_radiance(pixel_index, vec4<f32>(0.0));
    store_ray_samples(pixel_index, RaySamples(vec4<f32>(0.0), vec4<f32>(0.0)));
    surfaces[pixel_index].hit = 0u;
    surfaces[pixel_index].flags = 0u;
}
