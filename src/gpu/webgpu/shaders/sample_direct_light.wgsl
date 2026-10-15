@compute @workgroup_size(64, 1, 1)
fn sample_direct_light(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= direct_eval_count()) { return; }
    let ray_index = load_direct_eval_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let light_sample_origin = vec3<f32>(
        surface.light_sample_origin_x, surface.light_sample_origin_y, surface.light_sample_origin_z,
    );
    direct_light_samples[pixel_index] = sample_direct_light_at(
        surface.position.xyz,
        surface.normal.xyz,
        light_sample_origin,
        load_sample_lambda(pixel_index),
        load_ray_samples(pixel_index),
    );
}
