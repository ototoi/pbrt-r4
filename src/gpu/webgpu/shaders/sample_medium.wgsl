@compute @workgroup_size(64, 1, 1)
fn sample_medium(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= atomicLoad(&queue_counters.medium_active.count)) { return; }
    let ray_index = active_medium_indices[queue_index];
    var ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let infinite = surface.hit == 0u;
    let distance = select(surface.t * length(ray.direction.xyz), 0.0, infinite);
    let segment_distance = select(distance, RAY_T_MAX, infinite);
    var weight = vec4<f32>(1.0);

    if (ray.medium_id != 0xffffffffu) {
        if (ray.medium_id >= arrayLength(&media)) {
            set_render_error();
            return;
        }
        let medium = media[ray.medium_id];
        if (medium.kind != 0u) {
            set_render_error();
            return;
        }
        let lambda = load_sample_lambda(pixel_index);
        let sigma_a = evaluate_spectrum(medium.sigma_a, lambda);
        let sigma_s = evaluate_spectrum(medium.sigma_s, lambda);
        let sigma_t = sigma_a + sigma_s;
        if (any(sigma_a < vec4<f32>(0.0)) || any(sigma_s < vec4<f32>(0.0))
            || any(sigma_t != sigma_t) || any(abs(sigma_t) > vec4<f32>(RAY_T_MAX))) {
            set_render_error();
            return;
        }
        if (sigma_t.x > 0.0) {
            let u_distance = select(
                random01(pixel_index, 16384u + ray.medium_segment_index, ray.depth),
                random_medium(pixel_index, ray.medium_segment_index, ray.depth, 0u),
                viewport.medium_scattering_enabled != 0u,
            );
            let event_distance = -log(1.0 - u_distance) / sigma_t.x;
            if (event_distance < segment_distance) {
                let u_event = random_medium(pixel_index, ray.medium_segment_index, ray.depth, 2u);
                let p_absorb = sigma_a.x / sigma_t.x;
                let p_scatter = sigma_s.x / sigma_t.x;
                let p_null = max(0.0, 1.0 - p_absorb - p_scatter);
                let probability_sum = p_absorb + p_scatter + p_null;
                var event_sample = u_event * probability_sum;
                if (event_sample == probability_sum) {
                    event_sample = next_float_down(event_sample);
                }
                if (event_sample < p_absorb
                    || event_sample >= p_absorb + p_scatter
                    || ray.depth >= viewport.max_depth) {
                    ray.beta = vec4<f32>(0.0);
                    surfaces[pixel_index].hit = 0u;
                    store_current_ray(ray_index, ray);
                    return;
                }
                let scatter_weight = (sigma_s / sigma_s.x)
                    * exp((vec4<f32>(sigma_t.x) - sigma_t) * event_distance);
                ray.beta *= scatter_weight;
                ray.r_u *= scatter_weight;
                if (any(ray.beta != ray.beta) || any(ray.r_u != ray.r_u)
                    || any(abs(ray.beta) > vec4<f32>(RAY_T_MAX)) || any(abs(ray.r_u) > vec4<f32>(RAY_T_MAX))) {
                    set_render_error();
                    return;
                }
                let event_position = ray.origin.xyz
                    + normalize(ray.direction.xyz) * event_distance;
                if (any(event_position != event_position)
                    || any(abs(event_position) > vec3<f32>(RAY_T_MAX))) {
                    set_render_error();
                    return;
                }
                ray.origin = vec4<f32>(event_position, 1.0);
                surfaces[pixel_index].hit = 0u;
                store_current_ray(ray_index, ray);
                if (any(ray.beta != vec4<f32>(0.0)) && any(ray.r_u != vec4<f32>(0.0))) {
                    append_medium_scatter(ray_index);
                }
                return;
            }
            if (infinite) {
                weight = vec4<f32>(0.0);
            } else {
                weight = exp((vec4<f32>(sigma_t.x) - sigma_t) * distance);
            }
        } else if (sigma_t.x == 0.0) {
            if (infinite) {
                weight = select(vec4<f32>(0.0), vec4<f32>(1.0), sigma_t == vec4<f32>(0.0));
            } else {
                weight = exp(-distance * sigma_t);
            }
        } else {
            set_render_error();
            return;
        }
        ray.beta *= weight;
        ray.r_u *= weight;
        ray.r_l *= weight;
        if (any(ray.beta != ray.beta) || any(ray.r_u != ray.r_u) || any(ray.r_l != ray.r_l)
            || any(abs(ray.beta) > vec4<f32>(RAY_T_MAX)) || any(abs(ray.r_u) > vec4<f32>(RAY_T_MAX))
            || any(abs(ray.r_l) > vec4<f32>(RAY_T_MAX))) {
            set_render_error();
            return;
        }
    }

    if (!any(ray.beta != vec4<f32>(0.0))) {
        surfaces[pixel_index].hit = 0u;
        store_current_ray(ray_index, ray);
        return;
    }

    if (surface.hit == 1u
        && instances[surface.instance_custom_data].material_root == 0xffffffffu) {
        let triangle = reconstruct_triangle_surface(
            surface.instance_custom_data,
            surface.primitive_index,
            vec3<f32>(1.0 - surface.barycentric.x - surface.barycentric.y,
                      surface.barycentric.x, surface.barycentric.y),
        );
        if (triangle.valid == 0u) { return; }
        var boundary = surface;
        boundary.position = vec4<f32>(triangle.position, 1.0);
        boundary.position_error = vec4<f32>(triangle.position_error, 0.0);
        boundary.geometric_normal = vec4<f32>(triangle.geometric_normal, 0.0);
        let segment_progress = dot(triangle.position - ray.origin.xyz, ray.direction.xyz);
        if (!(segment_progress > 0.0)) {
            set_render_error();
            return;
        }
        ray.medium_id = interaction_get_medium(ray, boundary, ray.direction.xyz);
        ray.origin = vec4<f32>(offset_ray_origin(
            triangle.position, triangle.position_error,
            triangle.geometric_normal, ray.direction.xyz,
        ), 1.0);
        ray.medium_segment_index += 1u;
        surfaces[pixel_index].hit = 0u;
        append_medium_continuation(ray_index);
    }
    store_current_ray(ray_index, ray);
    if (infinite && any(ray.beta != vec4<f32>(0.0))) {
        append_escaped_ray(ray_index);
    }
}
