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
    let light_handle = instance.area_light;
    let area_light = load_light_payload(light_handle);
    let ray_index = find_current_ray_for_pixel(pixel_index);
    if (ray_index == 0xffffffffu) {
        return;
    }
    let ray = load_current_ray(ray_index);
    if (!load_area_two_sided(area_light)
        && dot(surface.geometric_normal.xyz, -ray.direction.xyz) <= 0.0) {
        return;
    }
    var weight = 1.0;
    if (ray.depth > 0u && ray.prev_pdf > 0.0) {
        var triangle_distribution_index = 0xffffffffu;
        let distribution_count = load_area_distribution_count(area_light);
        for (var i = 0u; i < distribution_count; i++) {
            if (load_area_distribution(area_light, i).primitive == surface.primitive_index) {
                triangle_distribution_index = i;
                break;
            }
        }
        if (triangle_distribution_index == 0xffffffffu) {
            return;
        }
        let triangle_selection = load_area_distribution(area_light, triangle_distribution_index);
        let triangle = load_area_triangle(area_light, triangle_selection.primitive);
        let triangle_pdf = uniform_triangle_pdf_for_context(
            triangle,
            ray.prev_position.xyz,
            surface.geometric_normal.xyz,
            ray.direction.xyz,
            surface.position.xyz,
            triangle_selection.area,
        );
        let light_pdf = light_pmf_for_handle(light_handle, ray.prev_position.xyz, ray.prev_shading_normal.xyz)
            * triangle_selection.pmf * triangle_pdf;
        weight = ray.prev_pdf / max(ray.prev_pdf + light_pdf, 1e-7);
    }
    store_sample_radiance(pixel_index, load_sample_radiance(pixel_index)
        + ray.throughput * load_area_emission(area_light) * weight);
}
