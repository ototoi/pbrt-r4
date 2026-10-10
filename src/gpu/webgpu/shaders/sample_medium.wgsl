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
        if (medium.kind > 1u) {
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
        if (medium.kind == 1u) {
            if (medium.grid_index >= arrayLength(&uniform_grid_media)) {
                set_render_error();
                return;
            }
            let grid = uniform_grid_media[medium.grid_index];
            let direction_world = normalize(ray.direction.xyz);
            var dda = uniform_grid_dda_init(
                grid, ray.origin.xyz, direction_world, 0.0, segment_distance,
            );
            var log_transmittance_maj = vec4<f32>(0.0);
            var candidate_index = 0u;
            var scattered = false;
            var absorbed = false;
            while (dda.valid != 0u && dda.t_min < dda.t_max) {
                let segment = uniform_grid_dda_next(grid, sigma_t, &dda);
                if (!(segment.t_max > segment.t_min)) { continue; }
                if (any(segment.sigma_maj != segment.sigma_maj)
                    || any(abs(segment.sigma_maj) > vec4<f32>(RAY_T_MAX))) {
                    set_render_error();
                    return;
                }
                if (segment.sigma_maj.x > 0.0) {
                    var t = segment.t_min;
                    loop {
                        let u = random_medium(
                            pixel_index, ray.medium_segment_index, ray.depth,
                            0x10000u + candidate_index * 2u,
                        );
                        candidate_index += 1u;
                        let candidate_t = t - log(1.0 - u) / segment.sigma_maj.x;
                        if (!(candidate_t < segment.t_max)) {
                            log_transmittance_maj -= (segment.t_max - t) * segment.sigma_maj;
                            break;
                        }
                        log_transmittance_maj -= (candidate_t - t) * segment.sigma_maj;
                        if (any(log_transmittance_maj != log_transmittance_maj)
                            || any(abs(log_transmittance_maj) > vec4<f32>(RAY_T_MAX))) {
                            set_render_error();
                            return;
                        }
                        let transmittance_ratio = exp(
                            log_transmittance_maj - vec4<f32>(log_transmittance_maj.x),
                        );
                        let position_world = ray.origin.xyz + direction_world * candidate_t;
                        let position_medium = (grid.medium_from_world
                            * vec4<f32>(position_world, 1.0)).xyz;
                        let density = uniform_grid_density(grid, position_medium);
                        let local_sigma_a = sigma_a * density;
                        let local_sigma_s = sigma_s * density;
                        if (any(local_sigma_a != local_sigma_a)
                            || any(local_sigma_s != local_sigma_s)
                            || any(abs(local_sigma_a) > vec4<f32>(RAY_T_MAX))
                            || any(abs(local_sigma_s) > vec4<f32>(RAY_T_MAX))) {
                            set_render_error();
                            return;
                        }
                        let local_sigma_n = max(
                            vec4<f32>(0.0), segment.sigma_maj - local_sigma_a - local_sigma_s,
                        );
                        let p_absorb = local_sigma_a.x / segment.sigma_maj.x;
                        let p_scatter = local_sigma_s.x / segment.sigma_maj.x;
                        let p_null = max(0.0, 1.0 - p_absorb - p_scatter);
                        let u_event = random_medium(
                            pixel_index, ray.medium_segment_index, ray.depth,
                            0x10001u + (candidate_index - 1u) * 2u,
                        );
                        let probability_sum = p_absorb + p_scatter + p_null;
                        var event_sample = u_event * probability_sum;
                        if (event_sample == probability_sum) {
                            event_sample = next_float_down(event_sample);
                        }
                        if (event_sample < p_absorb) {
                            absorbed = true;
                            break;
                        }
                        if (event_sample < p_absorb + p_scatter) {
                            if (!(local_sigma_s.x > 0.0)) {
                                set_render_error();
                                return;
                            }
                            let scatter_weight = transmittance_ratio
                                * local_sigma_s / local_sigma_s.x;
                            ray.beta *= scatter_weight;
                            ray.r_u *= scatter_weight;
                            let event_position = position_world;
                            if (any(event_position != event_position)
                                || any(abs(event_position) > vec3<f32>(RAY_T_MAX))) {
                                set_render_error();
                                return;
                            }
                            ray.origin = vec4<f32>(event_position, 1.0);
                            scattered = true;
                            break;
                        }
                        if (!(local_sigma_n.x > 0.0) || p_null <= 0.0) {
                            set_render_error();
                            return;
                        }
                        ray.beta *= transmittance_ratio * local_sigma_n / local_sigma_n.x;
                        ray.r_u *= transmittance_ratio * local_sigma_n / local_sigma_n.x;
                        ray.r_l *= transmittance_ratio * segment.sigma_maj / local_sigma_n.x;
                        if (any(ray.beta != ray.beta) || any(ray.r_u != ray.r_u)
                            || any(ray.r_l != ray.r_l)
                            || any(abs(ray.beta) > vec4<f32>(RAY_T_MAX))
                            || any(abs(ray.r_u) > vec4<f32>(RAY_T_MAX))
                            || any(abs(ray.r_l) > vec4<f32>(RAY_T_MAX))) {
                            set_render_error();
                            return;
                        }
                        log_transmittance_maj = vec4<f32>(0.0);
                        t = candidate_t;
                    }
                } else {
                    log_transmittance_maj -=
                        (segment.t_max - segment.t_min) * segment.sigma_maj;
                }
                if (any(log_transmittance_maj != log_transmittance_maj)
                    || any(abs(log_transmittance_maj) > vec4<f32>(RAY_T_MAX))) {
                    set_render_error();
                    return;
                }
                if (scattered || absorbed) { break; }
            }
            if (absorbed) {
                ray.beta = vec4<f32>(0.0);
                surfaces[pixel_index].hit = 0u;
                store_current_ray(ray_index, ray);
                return;
            }
            if (scattered) {
                if (ray.depth >= viewport.max_depth) {
                    ray.beta = vec4<f32>(0.0);
                    surfaces[pixel_index].hit = 0u;
                    store_current_ray(ray_index, ray);
                    return;
                }
                surfaces[pixel_index].hit = 0u;
                store_current_ray(ray_index, ray);
                if (any(ray.beta != vec4<f32>(0.0)) && any(ray.r_u != vec4<f32>(0.0))) {
                    append_medium_scatter(ray_index);
                }
                return;
            }
            weight = exp(log_transmittance_maj - vec4<f32>(log_transmittance_maj.x));
            ray.beta *= weight;
            ray.r_u *= weight;
            ray.r_l *= weight;
            if (any(ray.beta != ray.beta) || any(ray.r_u != ray.r_u)
                || any(ray.r_l != ray.r_l)
                || any(abs(ray.beta) > vec4<f32>(RAY_T_MAX))
                || any(abs(ray.r_u) > vec4<f32>(RAY_T_MAX))
                || any(abs(ray.r_l) > vec4<f32>(RAY_T_MAX))) {
                set_render_error();
                return;
            }
        } else {
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
