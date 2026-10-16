@compute @workgroup_size(64)
fn select_surface_light(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= direct_eval_count()) { return; }
    let ray_index = load_direct_eval_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    select_and_enqueue_light_sample(
        ray_index, pixel_index, LIGHT_SAMPLE_SOURCE_SURFACE,
        surface.position.xyz, surface.normal.xyz,
    );
}
