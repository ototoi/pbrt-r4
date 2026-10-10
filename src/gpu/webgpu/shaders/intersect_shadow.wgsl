@compute @workgroup_size(64, 1, 1)
fn intersect_shadow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= atomicLoad(&queue_counters.shadow_active.count)) {
        return;
    }
    let ray_index = active_shadow_indices[queue_index];
    var shadow = shadow_rays[ray_index];

    let shadow_origin = shadow.origin.xyz;
    let shadow_direction = shadow.direction.xyz;
    let shadow_t = shadow.max_t;
    if (shadow_t <= 0.0) {
        shadow_rays[ray_index] = shadow;
        return;
    }
    var query: ray_query;
    rayQueryInitialize(
        &query,
        tlas,
        RayDesc(0u, 0xffu, 0.0, shadow_t, shadow_origin, shadow_direction),
    );
    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        if (candidate.kind == RAY_QUERY_INTERSECTION_TRIANGLE
            && alpha_mask_candidate_accept(shadow_origin, shadow_direction, candidate)) {
            rayQueryConfirmIntersection(&query);
        }
    }
    let intersection = rayQueryGetCommittedIntersection(&query);
    let hit = intersection.kind != RAY_QUERY_INTERSECTION_NONE;
    let segment_distance = select(
        shadow_t * length(shadow_direction),
        intersection.t * length(shadow_direction),
        hit,
    );

    if (shadow.medium_id != 0xffffffffu) {
        if (shadow.medium_id >= arrayLength(&media)) {
            set_render_error();
            return;
        }
        let medium = media[shadow.medium_id];
        if (medium.kind > 1u) {
            set_render_error();
            return;
        }
        let lambda = load_sample_lambda(shadow.pixel_index);
        let sigma_a = evaluate_spectrum(medium.sigma_a, lambda);
        let sigma_s = evaluate_spectrum(medium.sigma_s, lambda);
        let sigma_t = sigma_a + sigma_s;
        var weight = vec4<f32>(1.0);
        if (any(sigma_a < vec4<f32>(0.0)) || any(sigma_s < vec4<f32>(0.0))
            || any(sigma_t != sigma_t) || any(abs(sigma_t) > vec4<f32>(RAY_T_MAX))) {
            set_render_error();
            return;
        }
        let infinite_segment = shadow.infinite_distance != 0u && !hit;
        if (medium.kind == 1u) {
            if (medium.grid_index >= arrayLength(&uniform_grid_media)) {
                set_render_error();
                return;
            }
            let grid = uniform_grid_media[medium.grid_index];
            let direction_world = normalize(shadow_direction);
            var dda = uniform_grid_dda_init(
                grid, shadow_origin, direction_world, 0.0,
                select(segment_distance, RAY_T_MAX, infinite_segment),
            );
            var log_transmittance_maj = vec4<f32>(0.0);
            var candidate_index = 0u;
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
                            shadow.pixel_index, shadow.segment_index, shadow.depth,
                            0x20000u + candidate_index,
                        );
                        candidate_index += 1u;
                        let candidate_t = t - log(1.0 - u) / segment.sigma_maj.x;
                        if (!(candidate_t < segment.t_max)) {
                            log_transmittance_maj -=
                                (segment.t_max - t) * segment.sigma_maj;
                            break;
                        }
                        log_transmittance_maj -=
                            (candidate_t - t) * segment.sigma_maj;
                        if (any(log_transmittance_maj != log_transmittance_maj)
                            || any(abs(log_transmittance_maj) > vec4<f32>(RAY_T_MAX))) {
                            set_render_error();
                            return;
                        }
                        let transmittance_ratio = exp(
                            log_transmittance_maj - vec4<f32>(log_transmittance_maj.x),
                        );
                        let position_world = shadow_origin + direction_world * candidate_t;
                        let position_medium = (grid.medium_from_world
                            * vec4<f32>(position_world, 1.0)).xyz;
                        let density = uniform_grid_density(grid, position_medium);
                        let local_sigma_t = sigma_t * density;
                        if (any(local_sigma_t != local_sigma_t)
                            || any(abs(local_sigma_t) > vec4<f32>(RAY_T_MAX))) {
                            set_render_error();
                            return;
                        }
                        let sigma_n = max(
                            vec4<f32>(0.0), segment.sigma_maj - local_sigma_t,
                        );
                        if (!(segment.sigma_maj.x > 0.0)) {
                            set_render_error();
                            return;
                        }
                        shadow.transmittance *= transmittance_ratio * sigma_n
                            / segment.sigma_maj.x;
                        shadow.inv_w_u *= transmittance_ratio * sigma_n
                            / segment.sigma_maj.x;
                        shadow.inv_w_l *= transmittance_ratio * segment.sigma_maj
                            / segment.sigma_maj.x;
                        log_transmittance_maj = vec4<f32>(0.0);
                        if (any(shadow.transmittance != shadow.transmittance)
                            || any(shadow.inv_w_u != shadow.inv_w_u)
                            || any(shadow.inv_w_l != shadow.inv_w_l)
                            || any(abs(shadow.transmittance) > vec4<f32>(RAY_T_MAX))
                            || any(abs(shadow.inv_w_u) > vec4<f32>(RAY_T_MAX))
                            || any(abs(shadow.inv_w_l) > vec4<f32>(RAY_T_MAX))) {
                            set_render_error();
                            return;
                        }
                        let roulette_denominator = sampled_spectrum_average(
                            shadow.inv_w_l + shadow.inv_w_u,
                        );
                        if (roulette_denominator > 0.0) {
                            let tr = shadow.transmittance / roulette_denominator;
                            if (max(max(tr.x, tr.y), max(tr.z, tr.w)) < 0.05) {
                                let roulette = random_medium(
                                    shadow.pixel_index, shadow.segment_index, shadow.depth,
                                    0x30000u + candidate_index,
                                );
                                if (roulette < 0.75) {
                                    shadow.transmittance = vec4<f32>(0.0);
                                } else {
                                    shadow.transmittance /= 0.25;
                                }
                            }
                        }
                        if (!any(shadow.transmittance != vec4<f32>(0.0))) { break; }
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
                if (!any(shadow.transmittance != vec4<f32>(0.0))) { break; }
            }
            let weight = exp(log_transmittance_maj - vec4<f32>(log_transmittance_maj.x));
            shadow.transmittance *= weight;
            shadow.inv_w_u *= weight;
            shadow.inv_w_l *= weight;
            if (any(shadow.transmittance != shadow.transmittance)
                || any(shadow.inv_w_u != shadow.inv_w_u)
                || any(shadow.inv_w_l != shadow.inv_w_l)
                || any(abs(shadow.transmittance) > vec4<f32>(RAY_T_MAX))
                || any(abs(shadow.inv_w_u) > vec4<f32>(RAY_T_MAX))
                || any(abs(shadow.inv_w_l) > vec4<f32>(RAY_T_MAX))) {
                set_render_error();
                return;
            }
        } else if (sigma_t.x > 0.0) {
            let u_distance = select(
                random01_stream(
                    shadow.pixel_index, 16384u + shadow.segment_index, shadow.depth, 1u,
                ),
                random_medium(shadow.pixel_index, shadow.segment_index, shadow.depth, 1u),
                viewport.medium_scattering_enabled != 0u,
            );
            let event_distance = -log(1.0 - u_distance) / sigma_t.x;
            if (event_distance < segment_distance) {
                shadow.transmittance = vec4<f32>(0.0);
                shadow_rays[ray_index] = shadow;
                return;
            }
            if (infinite_segment) {
                weight = vec4<f32>(0.0);
            } else {
                weight = exp((vec4<f32>(sigma_t.x) - sigma_t) * segment_distance);
            }
        } else if (sigma_t.x == 0.0) {
            if (infinite_segment) {
                weight = select(vec4<f32>(0.0), vec4<f32>(1.0), sigma_t == vec4<f32>(0.0));
            } else {
                weight = exp(-segment_distance * sigma_t);
            }
        } else {
            set_render_error();
            return;
        }
        shadow.transmittance *= weight;
        shadow.inv_w_u *= weight;
        shadow.inv_w_l *= weight;
        if (any(shadow.transmittance != shadow.transmittance)
            || any(shadow.inv_w_u != shadow.inv_w_u)
            || any(shadow.inv_w_l != shadow.inv_w_l)
            || any(abs(shadow.transmittance) > vec4<f32>(RAY_T_MAX))
            || any(abs(shadow.inv_w_u) > vec4<f32>(RAY_T_MAX))
            || any(abs(shadow.inv_w_l) > vec4<f32>(RAY_T_MAX))) {
            set_render_error();
            return;
        }
    }

    if (hit) {
        let instance = instances[intersection.instance_custom_data];
        if (instance.material_root != 0xffffffffu) {
            shadow.transmittance = vec4<f32>(0.0);
            shadow_rays[ray_index] = shadow;
            return;
        }

        let triangle = reconstruct_triangle_surface(
            intersection.instance_custom_data,
            intersection.primitive_index,
            vec3<f32>(1.0 - intersection.barycentrics.x - intersection.barycentrics.y,
                      intersection.barycentrics.x, intersection.barycentrics.y),
        );
        if (triangle.valid == 0u) {
            return;
        }
        if (!(dot(triangle.position - shadow_origin, shadow_direction) > 0.0)) {
            set_render_error();
            return;
        }
        shadow.origin = vec4<f32>(offset_ray_origin(
            triangle.position, triangle.position_error,
            triangle.geometric_normal, shadow_direction,
        ), 0.0);
        if (shadow.infinite_distance == 0u) {
            let to_endpoint = shadow.endpoint.xyz - shadow.origin.xyz;
            shadow.max_t = length(to_endpoint);
            shadow.direction = vec4<f32>(normalize(to_endpoint), 0.0);
        }
        if (instance.medium_inside != 0xffffffffu || instance.medium_outside != 0xffffffffu) {
            shadow.medium_id = select(
                instance.medium_inside,
                instance.medium_outside,
                dot(shadow.direction.xyz, triangle.geometric_normal) > 0.0,
            );
        }
        shadow.segment_index += 1u;
        shadow_rays[ray_index] = shadow;
        append_shadow_continuation(ray_index);
        return;
    }

    let denom = sampled_spectrum_average(shadow.r_u * shadow.inv_w_u + shadow.r_l * shadow.inv_w_l);
    if (denom != denom || abs(denom) > RAY_T_MAX) {
        set_render_error();
        return;
    }
    if (denom > 0.0 && any(shadow.transmittance != vec4<f32>(0.0))) {
        store_sample_radiance(
            shadow.pixel_index,
            load_sample_radiance(shadow.pixel_index)
                + shadow.direct * shadow.transmittance / denom,
        );
    }
    shadow_rays[ray_index] = shadow;
}
