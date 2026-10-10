@compute @workgroup_size(64, 1, 1)
fn handle_escaped_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
                let uv = portal_image_infinite_light_image_from_render(portal, normalize(ray.direction.xyz));
                let bounds = portal_image_infinite_light_image_bounds(portal, ray.origin.xyz);
                if (uv.valid != 0u && bounds.valid != 0u && all(uv.uv >= bounds.min) && all(uv.uv <= bounds.max)) {
                    light_radiance = load_portal_image_spectrum(light_index, uv.uv, lambda)
                        * load_light_scale(light_index);
                }
                light_pdf = light_sampler_pmf(
                    light_index, ray.prev_position.xyz, ray.prev_shading_normal.xyz,
                ) * portal_image_infinite_light_pdf_li(portal, uv, ray.prev_position.xyz);
            } else if (light_kind == LIGHT_KIND_IMAGE_INFINITE) {
                light_radiance = load_light_image_spectrum(light_index, ray.direction.xyz, lambda)
                    * load_light_scale(light_index);
                let model = light_sampling_models[load_light_payload(light_index)];
                if (model.geometry_kind != LIGHT_GEOMETRY_KIND_IMAGE_INFINITE
                    || model.geometry_index >= arrayLength(&image_infinite_sampling_records)) {
                    set_render_error();
                    continue;
                }
                let image = image_infinite_sampling_records[model.geometry_index];
                let map_uv = equal_area_sphere_to_square(normalize(vec3<f32>(
                    dot(model.world_to_light0.xyz, ray.direction.xyz),
                    dot(model.world_to_light1.xyz, ray.direction.xyz),
                    dot(model.world_to_light2.xyz, ray.direction.xyz),
                )));
                let map_pdf = piecewise_constant_2d_pdf(image, map_uv);
                let local_direction = normalize(vec3<f32>(
                    dot(model.world_to_light0.xyz, ray.direction.xyz),
                    dot(model.world_to_light1.xyz, ray.direction.xyz),
                    dot(model.world_to_light2.xyz, ray.direction.xyz),
                ));
                let jacobian = image_infinite_direction_jacobian(image, local_direction);
                if (!(jacobian > 0.0)) {
                    set_render_error();
                    continue;
                }
                light_pdf = light_sampler_pmf(
                    light_index, ray.prev_position.xyz, ray.prev_shading_normal.xyz,
                ) * map_pdf / (4.0 * PI * jacobian);
            } else {
                light_radiance = load_light_spectrum(light_index, 0u, lambda)
                    * load_light_scale(light_index);
                light_pdf = light_sampler_pmf(
                    light_index, ray.prev_position.xyz, ray.prev_shading_normal.xyz,
                ) / (4.0 * PI);
            }
            // Balance-heuristic MIS with the v4 rescaled-path-probability
            // (r_u/r_l) formulation: depth==0 or a specular previous bounce
            // means this light could only ever be found this way (no MIS
            // against BSDF sampling); otherwise weight by the ratio between
            // the BSDF-sampling density (r_u) and the light-sampling density
            // (r_l * light_pdf).
            if (ray.depth == 0u || ray.prev_specular != 0u) {
                let denom = sampled_spectrum_average(ray.r_u);
                if (denom > 1e-7) {
                    radiance += light_radiance / denom;
                }
            } else {
                let r_l = ray.r_l * light_pdf;
                let denom = sampled_spectrum_average(ray.r_u + r_l);
                if (denom > 1e-7) {
                    radiance += light_radiance / denom;
                }
            }
        }
    }
    store_sample_radiance(
        pixel_index,
        load_sample_radiance(pixel_index) + ray.beta * radiance,
    );
}
