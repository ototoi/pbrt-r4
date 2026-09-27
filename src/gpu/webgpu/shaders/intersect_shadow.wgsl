@compute @workgroup_size(64, 1, 1)
fn intersect_shadow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (ray_index >= shadow_ray_count()) {
        return;
    }
    var shadow = shadow_rays[ray_index];
    if (shadow.status == 1u) {
        return;
    }

    let shadow_origin = shadow.origin.xyz;
    let shadow_direction = shadow.direction.xyz;
    let shadow_t = shadow.max_t;
    if (shadow_t <= 0.0) {
        shadow.status = 1u;
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
        if (medium.kind != 0u) {
            set_render_error();
            return;
        }
        let sigma_a = evaluate_spectrum(medium.sigma_a, load_sample_lambda(shadow.pixel_index));
        var weight = vec4<f32>(1.0);
        if (sigma_a.x > 0.0) {
            if (shadow.infinite_distance != 0u && !hit) {
                shadow.transmittance = vec4<f32>(0.0);
                shadow.status = 1u;
                shadow_rays[ray_index] = shadow;
                return;
            }
            let u = random01(shadow.pixel_index, 16385u + shadow.segment_index, shadow.depth);
            let event_distance = -log(1.0 - u) / sigma_a.x;
            if (event_distance < segment_distance) {
                shadow.transmittance = vec4<f32>(0.0);
                shadow.status = 1u;
                shadow_rays[ray_index] = shadow;
                return;
            }
            weight = exp((vec4<f32>(sigma_a.x) - sigma_a) * segment_distance);
        } else if (sigma_a.x == 0.0) {
            if (shadow.infinite_distance != 0u && !hit) {
                weight = select(vec4<f32>(0.0), vec4<f32>(1.0), sigma_a == vec4<f32>(0.0));
            } else {
                weight = exp(-segment_distance * sigma_a);
            }
        } else {
            set_render_error();
            return;
        }
        shadow.transmittance *= weight;
        shadow.inv_w_u *= weight;
        shadow.inv_w_l *= weight;
    }

    if (hit) {
        let instance = instances[intersection.instance_custom_data];
        if (instance.material_root != 0xffffffffu) {
            shadow.transmittance = vec4<f32>(0.0);
            shadow.status = 1u;
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
        shadow.status = 2u;
        shadow_rays[ray_index] = shadow;
        append_shadow_continuation();
        return;
    }

    let denom = average_spectrum(shadow.r_u * shadow.inv_w_u + shadow.r_l * shadow.inv_w_l);
    if (denom > 0.0 && any(shadow.transmittance != vec4<f32>(0.0))) {
        store_sample_radiance(
            shadow.pixel_index,
            load_sample_radiance(shadow.pixel_index)
                + shadow.direct * shadow.transmittance / denom,
        );
    }
    shadow.status = 1u;
    shadow_rays[ray_index] = shadow;
}
