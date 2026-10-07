fn invalid_direct_light_sample() -> DirectLightSample {
    return DirectLightSample(
        vec4<f32>(0.0), vec4<f32>(0.0), vec3<f32>(0.0), 0u,
        vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0u,
    );
}

fn sample_direct_light_at(
    position: vec3<f32>,
    normal: vec3<f32>,
    light_sample_origin: vec3<f32>,
    lambda: vec4<f32>,
    samples: RaySamples,
) -> DirectLightSample {
    let light_selection = sample_scene_light(samples.direct.x, position, normal);
    if (light_selection.pmf <= 0.0 || light_selection.index == 0xffffffffu) {
        return invalid_direct_light_sample();
    }
    let light_index = light_selection.index;
    let light_kind = load_light_kind(light_index);
    let light_payload = load_light_payload(light_index);
    var light_position = vec3<f32>(0.0);
    var light_error = vec3<f32>(0.0);
    var light_normal = vec3<f32>(0.0);
    var light_radiance = vec4<f32>(0.0);
    var sampled_light_pdf = light_selection.pmf;
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
    } else if (is_infinite_light_kind(light_kind)) {
        if (light_kind == LIGHT_KIND_PORTAL_IMAGE_INFINITE) {
            // Selecting and sampling the portal here, in the same
            // invocation that selected it, avoids re-deriving the same
            // light_selection from samples.direct.x in a separate stage.
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
            let bounds = portal_image_bounds(portal, position);
            let portal_sample = sample_portal_distribution(
                portal, vec2<f32>(samples.direct.y, samples.direct.z), bounds,
            );
            if (portal_sample.valid == 0u) { return invalid_direct_light_sample(); }
            let direction = portal_render_from_image(portal, portal_sample.uv);
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
            let image_sample = sample_image_infinite_distribution(
                image, vec2<f32>(samples.direct.y, samples.direct.z),
            );
            if (image_sample.valid == 0u) { return invalid_direct_light_sample(); }
            let w_light = image_infinite_equal_area_square_to_sphere(image_sample.uv);
            wi = vec3<f32>(
                dot(image.light_to_render0.xyz, w_light),
                dot(image.light_to_render1.xyz, w_light),
                dot(image.light_to_render2.xyz, w_light),
            );
            sampled_light_pdf = sampled_light_pdf * image_sample.pdf / (4.0 * PI);
            if (!(sampled_light_pdf > 0.0) || sampled_light_pdf != sampled_light_pdf) {
                return invalid_direct_light_sample();
            }
            light_radiance = load_light_image_spectrum_uv(
                light_index, image_sample.uv, lambda,
            ) * load_light_scale(light_index);
        } else {
            wi = sample_uniform_infinite_direction(samples.direct.yz);
            sampled_light_pdf = sampled_light_pdf / (4.0 * PI);
            light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
        }
    } else if (light_kind == LIGHT_KIND_AREA) {
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
        light_normal = triangle_geometric_normal(triangle);
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
    let use_mis = (light_kind == LIGHT_KIND_AREA && !area_light_is_zero_alpha_sample_only(light_payload))
        || is_infinite_light_kind(light_kind);
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
