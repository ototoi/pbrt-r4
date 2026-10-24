fn invalid_direct_light_sample() -> DirectLightSample {
    return DirectLightSample(
        vec4<f32>(0.0), vec4<f32>(0.0), vec3<f32>(0.0), 0u,
        vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0u,
    );
}

fn select_and_enqueue_light_sample(
    ray_index: u32,
    pixel_index: u32,
    source_kind: u32,
    position: vec3<f32>,
    normal: vec3<f32>,
) {
    direct_light_samples[pixel_index] = invalid_direct_light_sample();
    let samples = load_ray_samples(pixel_index);
    let selection = light_sampler_sample(samples.direct.x, position, normal);
    if (selection.pmf <= 0.0 || selection.index == 0xffffffffu) { return; }
    append_light_sample_work(ray_index, source_kind, selection);
}

fn append_light_sample_work(ray_index: u32, source_kind: u32, selection: LightSelection) {
    let light_kind = load_light_kind(selection.index);
    var slot = 2u;
    if (light_kind == LIGHT_KIND_AREA) {
        slot = 0u;
    } else if (light_kind == LIGHT_KIND_IMAGE_INFINITE || light_kind == LIGHT_KIND_PORTAL_IMAGE_INFINITE) {
        slot = 1u;
    } else if (light_kind != LIGHT_KIND_POINT && light_kind != LIGHT_KIND_SPOT
        && light_kind != LIGHT_KIND_DISTANT && light_kind != LIGHT_KIND_UNIFORM_INFINITE) {
        return;
    }
    var index = 0u;
    var capacity = 0u;
    if (slot == 0u) {
        index = atomicAdd(&queue_counters.sample_area_light.count, 1u);
        capacity = queue_counters.sample_area_light.capacity;
    } else if (slot == 1u) {
        index = atomicAdd(&queue_counters.sample_image_infinite_light.count, 1u);
        capacity = queue_counters.sample_image_infinite_light.capacity;
    } else {
        index = atomicAdd(&queue_counters.sample_other_light.count, 1u);
        capacity = queue_counters.sample_other_light.capacity;
    }
    if (index >= capacity) {
        if (slot == 0u) { atomicStore(&queue_counters.sample_area_light.overflow, 1u); }
        else if (slot == 1u) { atomicStore(&queue_counters.sample_image_infinite_light.overflow, 1u); }
        else { atomicStore(&queue_counters.sample_other_light.overflow, 1u); }
        set_render_error();
        return;
    }
    light_sample_work[slot * capacity + index] = LightSampleWorkItem(
        ray_index, source_kind, selection.index, selection.pmf,
    );
}

fn sample_area_light_at(work: LightSampleWorkItem, light_sample_origin: vec3<f32>, lambda: vec4<f32>, samples: RaySamples) -> DirectLightSample {
    let light_index = work.light_index;
    let light_kind = load_light_kind(light_index);
    let light_payload = load_light_payload(light_index);
    var light_position = vec3<f32>(0.0);
    var light_error = vec3<f32>(0.0);
    var light_normal = vec3<f32>(0.0);
    var light_radiance = vec4<f32>(0.0);
    var sampled_light_pdf = work.selection_pmf;
    var wi = vec3<f32>(0.0);
    var distance_squared = 1.0;

        let total_area = load_area_total(light_payload);
        let distribution_count = load_area_distribution_count(light_payload);
        if (total_area <= 0.0 || distribution_count == 0u) {
            return invalid_direct_light_sample();
        }
        let triangle_selection = select_area_triangle(light_payload, samples.direct.y);
        let triangle = load_area_triangle(light_payload, triangle_selection.primitive);
        let triangle_sample = sample_uniform_triangle_for_context(
            triangle,
            light_sample_origin,
            vec2<f32>(triangle_selection.u_remapped, samples.direct.z),
            triangle_selection.area,
        );
        if (triangle_sample.w <= 0.0) {
            return invalid_direct_light_sample();
        }
        let b = triangle_sample.xyz;
        light_normal = triangle_geometric_normal(triangle, b);
        light_position = triangle.p0.xyz * b.x + triangle.p1.xyz * b.y + triangle.p2.xyz * b.z;
        if (!alpha_area_sample_accept(
            light_payload, triangle_selection.primitive, b, light_position, light_normal,
        )) {
            return invalid_direct_light_sample();
        }
        light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
        light_error = (abs(triangle.p0.xyz * b.x) + abs(triangle.p1.xyz * b.y)
            + abs(triangle.p2.xyz * b.z)) * gamma(6.0)
            + hardware_intersection_error(triangle.p0.xyz, triangle.p1.xyz, triangle.p2.xyz);
        let area_wi = normalize(light_position - light_sample_origin);
        let cosine_light = dot(light_normal, -area_wi);
        if (area_light_is_two_sided(light_payload)) {
            if (abs(cosine_light) == 0.0) {
                return invalid_direct_light_sample();
            }
        } else if (cosine_light <= 0.0) {
            return invalid_direct_light_sample();
        }
        sampled_light_pdf = sampled_light_pdf * triangle_selection.pmf * triangle_sample.w;

    if (light_kind != LIGHT_KIND_DISTANT && !is_infinite_light_kind(light_kind)) {
        let to_light = light_position - light_sample_origin;
        distance_squared = dot(to_light, to_light);
        if (distance_squared <= 0.0) {
            return invalid_direct_light_sample();
        }
        wi = to_light / sqrt(distance_squared);
    }
    let use_mis = !area_light_is_zero_alpha_sample_only(light_payload);
    return DirectLightSample(
        vec4<f32>(wi, sampled_light_pdf),
        light_radiance,
        light_position,
        light_kind,
        light_error,
        u32(use_mis),
        light_normal,
        1u,
    );
}

fn sample_image_infinite_light_at(work: LightSampleWorkItem, position: vec3<f32>, lambda: vec4<f32>, samples: RaySamples) -> DirectLightSample {
    let light_sample_origin = position;
    let light_index = work.light_index;
    let light_kind = load_light_kind(light_index);
    let light_payload = load_light_payload(light_index);
    var light_position = vec3<f32>(0.0);
    var light_error = vec3<f32>(0.0);
    var light_normal = vec3<f32>(0.0);
    var light_radiance = vec4<f32>(0.0);
    var sampled_light_pdf = work.selection_pmf;
    var wi = vec3<f32>(0.0);
    var distance_squared = 1.0;
    if (light_kind == LIGHT_KIND_PORTAL_IMAGE_INFINITE) {

            // The selector stage records the selected light so this stage only
            // needs to sample the already-routed portal work item.
            let model = light_sampling_models[light_payload];
            if (model.geometry_kind != 3u || model.geometry_index >= arrayLength(&portal_infinite_lights)) {
                set_render_error();
                return invalid_direct_light_sample();
            }
            let portal = portal_infinite_lights[model.geometry_index];
            if (portal.width == 0u || portal.height == 0u) {
                set_render_error();
                return invalid_direct_light_sample();
            }
            let bounds = portal_image_infinite_light_image_bounds(portal, position);
            let portal_sample = windowed_piecewise_constant_2d_sample(
                portal, vec2<f32>(samples.direct.y, samples.direct.z), bounds,
            );
            if (portal_sample.valid == 0u) { return invalid_direct_light_sample(); }
            let direction = portal_image_infinite_light_render_from_image(portal, portal_sample.uv);
            if (direction.valid == 0u || direction.duv_dw <= 0.0) { return invalid_direct_light_sample(); }
            let portal_pdf = sampled_light_pdf * portal_sample.pdf / direction.duv_dw;
            if (!portal_finite(portal_pdf) || portal_pdf == 0.0) { return invalid_direct_light_sample(); }
            wi = direction.wi;
            sampled_light_pdf = portal_pdf;
            light_radiance = load_portal_image_spectrum(
                light_index, portal_sample.uv, lambda,
            ) * load_light_scale(light_index);
            } else if (light_kind == LIGHT_KIND_IMAGE_INFINITE) {

            let model = light_sampling_models[light_payload];
            if (model.geometry_kind != LIGHT_GEOMETRY_KIND_IMAGE_INFINITE
                || model.geometry_index >= arrayLength(&image_infinite_sampling_records)) {
                set_render_error();
                return invalid_direct_light_sample();
            }
            let image = image_infinite_sampling_records[model.geometry_index];
            let image_sample = piecewise_constant_2d_sample(
                image, vec2<f32>(samples.direct.y, samples.direct.z),
            );
            if (image_sample.valid == 0u) { return invalid_direct_light_sample(); }
            let w_light = equal_area_square_to_sphere(image_sample.uv);
            let transformed = vec3<f32>(
                dot(image.light_to_render0.xyz, w_light),
                dot(image.light_to_render1.xyz, w_light),
                dot(image.light_to_render2.xyz, w_light),
            );
            let jacobian = image_infinite_direction_jacobian(image, w_light);
            if (!(jacobian > 0.0)) { return invalid_direct_light_sample(); }
            wi = normalize(transformed);
            sampled_light_pdf = sampled_light_pdf * image_sample.pdf / (4.0 * PI * jacobian);
            if (!(sampled_light_pdf > 0.0) || sampled_light_pdf != sampled_light_pdf) {
                return invalid_direct_light_sample();
            }
            light_radiance = load_light_image_spectrum_uv(
                light_index, image_sample.uv, lambda,
            ) * load_light_scale(light_index);
            } else {
        return invalid_direct_light_sample();
    }
    if (light_kind != LIGHT_KIND_DISTANT && !is_infinite_light_kind(light_kind)) {
        let to_light = light_position - light_sample_origin;
        distance_squared = dot(to_light, to_light);
        if (distance_squared <= 0.0) {
            return invalid_direct_light_sample();
        }
        wi = to_light / sqrt(distance_squared);
    }
    let use_mis = true;
    return DirectLightSample(
        vec4<f32>(wi, sampled_light_pdf),
        light_radiance,
        light_position,
        light_kind,
        light_error,
        u32(use_mis),
        light_normal,
        1u,
    );
}

fn sample_other_light_at(work: LightSampleWorkItem, light_sample_origin: vec3<f32>, lambda: vec4<f32>, samples: RaySamples) -> DirectLightSample {
    let light_index = work.light_index;
    let light_kind = load_light_kind(light_index);
    let light_payload = load_light_payload(light_index);
    var light_position = vec3<f32>(0.0);
    var light_error = vec3<f32>(0.0);
    var light_normal = vec3<f32>(0.0);
    var light_radiance = vec4<f32>(0.0);
    var sampled_light_pdf = work.selection_pmf;
    var wi = vec3<f32>(0.0);
    var distance_squared = 1.0;
    if (light_kind == LIGHT_KIND_POINT) {

        light_position = load_point_position(light_index);
        light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
        } else if (light_kind == LIGHT_KIND_SPOT) {

        light_position = load_point_position(light_index);
        let spot_w = normalize(light_sample_origin - light_position);
        let falloff = spot_falloff(light_index, spot_w);
        light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index) * falloff;
        } else if (light_kind == LIGHT_KIND_DISTANT) {

        wi = normalize(load_light_direction(light_index));
        light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
        } else if (light_kind == LIGHT_KIND_UNIFORM_INFINITE) {

            wi = sample_uniform_sphere(samples.direct.yz);
            sampled_light_pdf = sampled_light_pdf / (4.0 * PI);
            light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
    } else {
        return invalid_direct_light_sample();
    }
    if (light_kind != LIGHT_KIND_DISTANT && !is_infinite_light_kind(light_kind)) {
        let to_light = light_position - light_sample_origin;
        distance_squared = dot(to_light, to_light);
        if (distance_squared <= 0.0) {
            return invalid_direct_light_sample();
        }
        wi = to_light / sqrt(distance_squared);
    }
    if (light_kind == LIGHT_KIND_POINT || light_kind == LIGHT_KIND_SPOT) {
        light_radiance = light_radiance / distance_squared;
    }
    let use_mis = light_kind == LIGHT_KIND_UNIFORM_INFINITE;
    return DirectLightSample(
        vec4<f32>(wi, sampled_light_pdf),
        light_radiance,
        light_position,
        light_kind,
        light_error,
        u32(use_mis),
        light_normal,
        1u,
    );
}
