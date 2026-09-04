@compute @workgroup_size(8, 8, 1)
fn handle_emissive(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * viewport.width + global_id.x;
    if (queue_index >= hit_area_light_count()) {
        return;
    }
    let pixel_index = load_hit_area_pixel(queue_index);
    let surface = surfaces[pixel_index];
    let instance = instances[surface.instance_custom_data];
    if (instance.area_light == 0xffffffffu) {
        return;
    }
    let ray_index = find_current_ray_for_pixel(pixel_index);
    if (ray_index == 0xffffffffu) {
        return;
    }
    let ray = load_current_ray(ray_index);
    var weight = 1.0;
    if (ray.depth > 0u && ray.prev_pdf > 0.0) {
        let total_area = load_area_total(instance.area_light);
        let light_pdf = light_pmf_for_area(instance.area_light)
            * dot(ray.origin.xyz - surface.position.xyz, ray.origin.xyz - surface.position.xyz)
            / (max(abs(dot(surface.normal.xyz, -ray.direction.xyz)), 1e-7) * total_area);
        weight = ray.prev_pdf / max(ray.prev_pdf + light_pdf, 1e-7);
    }
    store_sample_radiance(pixel_index, load_sample_radiance(pixel_index)
        + ray.throughput * load_area_emission(instance.area_light) * weight);
}
