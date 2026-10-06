@compute @workgroup_size(64, 1, 1)
fn scatter_medium(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= medium_scatter_count()) { return; }
    let ray_index = load_medium_scatter_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let medium = media[ray.medium_id];
    let lambda = load_sample_lambda(pixel_index);
    let samples = load_ray_samples(pixel_index);
    let p = ray.origin.xyz;
    let wo = -normalize(ray.direction.xyz);
    let g = medium.g;

    let light_sample = sample_direct_light_at(
        p, vec3<f32>(0.0), p, lambda, samples,
    );
    if (light_sample.valid != 0u && light_sample.direction_pdf.w > 0.0) {
        let wi = light_sample.direction_pdf.xyz;
        let p_phase = hg_phase(dot(wo, wi), g);
        if (p_phase > 0.0 && p_phase == p_phase) {
            let light_pdf = light_sample.direction_pdf.w;
            let direct = ray.beta * p_phase * light_sample.radiance;
            let delta = light_sample.light_kind == LIGHT_KIND_POINT
                || light_sample.light_kind == LIGHT_KIND_SPOT
                || light_sample.light_kind == LIGHT_KIND_DISTANT;
            let r_u = select(ray.r_u * p_phase, vec4<f32>(0.0), delta);
            let r_l = ray.r_u * light_pdf;
            if (any(direct != direct) || any(r_u != r_u) || any(r_l != r_l)
                || any(abs(direct) > vec4<f32>(RAY_T_MAX)) || any(abs(r_u) > vec4<f32>(RAY_T_MAX))
                || any(abs(r_l) > vec4<f32>(RAY_T_MAX))) {
                set_render_error();
                return;
            }
            let infinite_distance = select(
                0u, 1u,
                light_sample.light_kind == LIGHT_KIND_DISTANT
                    || is_infinite_light_kind(light_sample.light_kind),
            );
            var shadow_direction = wi;
            var shadow_distance = RAY_T_MAX;
            if (light_sample.light_kind == LIGHT_KIND_AREA) {
                let shadow_target = offset_ray_origin(
                    light_sample.position, light_sample.position_error,
                    light_sample.normal, -wi,
                );
                let to_target = shadow_target - p;
                shadow_distance = length(to_target);
                if (!(shadow_distance > 0.0) || shadow_distance != shadow_distance
                    || shadow_distance > RAY_T_MAX) {
                    set_render_error();
                    return;
                }
                shadow_direction = to_target / shadow_distance;
            } else if (infinite_distance == 0u) {
                let to_light = light_sample.position - p;
                shadow_distance = length(to_light);
                shadow_direction = to_light / shadow_distance;
            }
            append_shadow_ray(
                pixel_index, p, shadow_direction, shadow_distance,
                ray.medium_id, ray.depth, infinite_distance, direct, r_u, r_l,
            );
        }
    }

    let wi = sample_hg_direction(wo, samples.indirect.yz, g);
    let phase_pdf = hg_phase(dot(wo, wi), g);
    if (!(phase_pdf > 0.0) || phase_pdf != phase_pdf) { return; }
    var next_beta = ray.beta;
    if (ray.depth >= 1u) {
        let average_r_u = average_spectrum(ray.r_u);
        if (!(average_r_u > 0.0) || average_r_u != average_r_u) { return; }
        let rr_beta = max_spectrum(next_beta * ray.eta_scale) / average_r_u;
        if (rr_beta != rr_beta || abs(rr_beta) > RAY_T_MAX) {
            set_render_error();
            return;
        }
        let q = clamp(1.0 - rr_beta, 0.0, 1.0);
        if (samples.indirect.w < q) { return; }
        let survival = 1.0 - q;
        if (!(survival > 0.0)) { return; }
        next_beta /= survival;
    }
    if (any(next_beta != next_beta) || any(abs(next_beta) > vec4<f32>(RAY_T_MAX))) {
        set_render_error();
        return;
    }
    let next_r_l = ray.r_u / phase_pdf;
    if (any(next_r_l != next_r_l) || any(abs(next_r_l) > vec4<f32>(RAY_T_MAX))) {
        set_render_error();
        return;
    }
    let next_ray = RayWorkItem(
        vec4<f32>(p, 1.0),
        vec4<f32>(wi, 0.0),
        next_beta,
        ray.r_u,
        next_r_l,
        vec4<f32>(p, 1.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        pixel_index,
        ray.depth + 1u,
        ray.eta_scale,
        phase_pdf,
        0u,
        ray.medium_id,
        0u,
        0u,
    );
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= queue_counters.next.capacity) {
        atomicStore(&queue_counters.next.overflow, 1u);
        return;
    }
    store_next_ray(next_index, next_ray);
}
