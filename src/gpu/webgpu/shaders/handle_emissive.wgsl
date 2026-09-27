@compute @workgroup_size(64, 1, 1)
fn handle_emissive(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= hit_area_light_count()) {
        return;
    }
    let ray_index = load_hit_area_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let instance = instances[surface.instance_custom_data];
    if (instance.area_light == 0xffffffffu) {
        return;
    }
    let light_handle = instance.area_light;
    let area_light = load_light_payload(light_handle);
    if (!area_light_is_zero_alpha_sample_only(area_light)) {
        let alpha = alpha_mask_value(
            instance.material_root,
            surface.uv,
            surface.position.xyz,
            surface.normal.xyz,
        );
        if (!alpha_mask_point_accept(alpha, surface.position.xyz)) {
            return;
        }
    }
    if (!area_light_is_two_sided(area_light)
        && dot(surface.geometric_normal.xyz, -ray.direction.xyz) <= 0.0) {
        return;
    }
    // Balance-heuristic MIS with the v4 rescaled-path-probability (r_u/r_l)
    // formulation; see handle_escaped.wgsl for the analogous infinite-light
    // case. depth==0 or a specular previous bounce means this light could
    // only ever be found this way (no MIS against BSDF sampling).
    var denom = 1.0;
    if (ray.depth == 0u || ray.prev_specular != 0u) {
        denom = average_spectrum(ray.r_u);
    } else {
        var triangle_distribution_index = 0xffffffffu;
        let distribution_count = load_area_distribution_count(area_light);
        if (distribution_count == 0u || load_area_total(area_light) <= 0.0) {
            return;
        }
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
        let r_l = ray.r_l * light_pdf;
        denom = average_spectrum(ray.r_u + r_l);
    }
    if (denom <= 1e-7) {
        return;
    }
    store_sample_radiance(pixel_index, load_sample_radiance(pixel_index)
        + ray.beta * load_light_spectrum(light_handle, 0u, load_sample_lambda(pixel_index))
            * load_light_scale(light_handle) / denom);
}
