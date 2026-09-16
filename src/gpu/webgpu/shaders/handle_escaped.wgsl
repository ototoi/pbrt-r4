@compute @workgroup_size(8, 8, 1)
fn handle_escaped(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * viewport.width + global_id.x;
    if (index >= escaped_ray_count()) {
        return;
    }
    let ray_index = escaped_ray_indices[index];
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let lambda = load_sample_lambda(pixel_index);
    var radiance = vec4<f32>(0.0);
    for (var light_index = 0u; light_index < light_table.light_count; light_index++) {
        let light_kind = load_light_kind(light_index);
        if (light_kind == LIGHT_KIND_UNIFORM_INFINITE
            || light_kind == LIGHT_KIND_IMAGE_INFINITE
            || light_kind == LIGHT_KIND_PORTAL_IMAGE_INFINITE) {
            radiance += load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
        }
    }
    store_sample_radiance(pixel_index, load_sample_radiance(pixel_index) + ray.throughput * radiance);
}
