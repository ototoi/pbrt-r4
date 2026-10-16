fn load_sample_area_light_work(index: u32) -> LightSampleWorkItem { return light_sample_work[0u * queue_counters.sample_area_light.capacity + index]; }

@compute @workgroup_size(64)
fn sample_area_light(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    let count = atomicLoad(&queue_counters.sample_area_light.count);
    if (index >= min(count, queue_counters.sample_area_light.capacity)) { return; }
    let work = load_sample_area_light_work(index);
    let ray = load_current_ray(work.ray_index);
    let pixel_index = ray.pixel_index;
    var light_sample_origin = ray.origin.xyz;
    if (work.source_kind == LIGHT_SAMPLE_SOURCE_SURFACE) {
        let surface = surfaces[pixel_index];
        light_sample_origin = vec3<f32>(surface.light_sample_origin_x, surface.light_sample_origin_y, surface.light_sample_origin_z);
    } else if (work.source_kind != LIGHT_SAMPLE_SOURCE_MEDIUM) {
        set_render_error();
        return;
    }
    direct_light_samples[pixel_index] = sample_area_light_at(work, light_sample_origin, load_sample_lambda(pixel_index), load_ray_samples(pixel_index));
}
