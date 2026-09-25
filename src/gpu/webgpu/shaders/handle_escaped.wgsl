@compute @workgroup_size(64, 1, 1)
fn handle_escaped(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
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
            var light_radiance = vec4<f32>(0.0);
            var light_pdf = 0.0;
            if (light_kind == LIGHT_KIND_PORTAL_IMAGE_INFINITE) {
                let model = light_sampling_models[load_light_payload(light_index)];
                if (model.geometry_kind != 3u
                    || model.geometry_index >= arrayLength(&portal_infinite_lights)) {
                    set_render_error();
                    continue;
                }
                let portal = portal_infinite_lights[model.geometry_index];
                let uv = portal_image_from_render(portal, normalize(ray.direction.xyz));
                let bounds = portal_image_bounds(portal, ray.origin.xyz);
                if (uv.valid != 0u && bounds.valid != 0u && all(uv.uv >= bounds.min) && all(uv.uv <= bounds.max)) {
                    light_radiance = load_portal_image_spectrum(light_index, uv.uv, lambda)
                        * load_light_scale(light_index);
                }
                light_pdf = light_pmf_for_handle(
                    light_index, ray.prev_position.xyz, ray.prev_shading_normal.xyz,
                ) * portal_pdf_li(portal, uv, ray.prev_position.xyz);
            } else if (light_kind == LIGHT_KIND_IMAGE_INFINITE) {
                light_radiance = load_light_image_spectrum(light_index, ray.direction.xyz, lambda)
                    * load_light_scale(light_index);
                light_pdf = light_pmf_for_handle(
                    light_index, ray.prev_position.xyz, ray.prev_shading_normal.xyz,
                ) / (4.0 * PI);
            } else {
                light_radiance = load_light_spectrum(light_index, 0u, lambda)
                    * load_light_scale(light_index);
                light_pdf = light_pmf_for_handle(
                    light_index, ray.prev_position.xyz, ray.prev_shading_normal.xyz,
                ) / (4.0 * PI);
            }
            var mis_weight = 1.0;
            if (ray.depth > 0u && ray.prev_specular == 0u && light_pdf > 0.0) {
                let bsdf_pdf = ray.prev_pdf;
                let bsdf_pdf2 = bsdf_pdf * bsdf_pdf;
                let light_pdf2 = light_pdf * light_pdf;
                mis_weight = bsdf_pdf2 / max(bsdf_pdf2 + light_pdf2, 1e-7);
            }
            radiance += light_radiance * mis_weight;
        }
    }
    store_sample_radiance(
        pixel_index,
        load_sample_radiance(pixel_index) + ray.throughput * radiance,
    );
}
