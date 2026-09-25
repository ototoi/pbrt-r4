@compute @workgroup_size(8, 8, 1)
fn sample_direct_light(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let queue_index = global_id.y * viewport.width + global_id.x;
    if (queue_index >= material_eval_count()) {
        return;
    }
    let ray_index = load_material_eval_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    direct_light_samples[pixel_index].valid = 0u;
    let surface = surfaces[pixel_index];
    let lambda = load_sample_lambda(pixel_index);
    let samples = load_ray_samples(pixel_index);
    let selected_evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);
    let root_surface_kind = selected_evaluated.bxdf_kind;
    var material_kind = root_surface_kind;
    if (root_surface_kind == MATERIAL_KIND_COATED_DIFFUSE) {
        material_kind = MATERIAL_KIND_DIFFUSE;
    } else if (root_surface_kind == MATERIAL_KIND_COATED_CONDUCTOR) {
        material_kind = MATERIAL_KIND_CONDUCTOR_ETA_K;
    }
    if (surface.hit == 0u
        || (material_kind != MATERIAL_KIND_DIFFUSE
            && material_kind != MATERIAL_KIND_DIFFUSE_TRANSMISSION
            && material_kind != MATERIAL_KIND_CONDUCTOR_ETA_K
            && material_kind != MATERIAL_KIND_CONDUCTOR_REFLECTANCE
            && material_kind != MATERIAL_KIND_MEASURED)) {
        return;
    }
    if (ray.depth >= viewport.max_depth || light_table.light_count == 0u) {
        return;
    }
    let wo = -ray.direction.xyz;
    // Flat IR represents the currently supported non-specular
    // `is_diffuse && is_transmission` combination as DiffuseTransmission.
    let light_sample_offset_direction = select(
        wo,
        -wo,
        material_kind == MATERIAL_KIND_DIFFUSE_TRANSMISSION,
    );
    let light_sample_origin = offset_ray_origin(
        surface.position.xyz,
        surface.position_error.xyz,
        surface.geometric_normal.xyz,
        light_sample_offset_direction,
    );
    let light_selection = sample_scene_light(samples.direct.x, surface.position.xyz, surface.normal.xyz);
    if (light_selection.pmf <= 0.0 || light_selection.index == 0xffffffffu) {
        return;
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
            let candidate = portal_light_candidates[pixel_index];
            if (candidate.state != PORTAL_CANDIDATE_SAMPLED
                || candidate.light_index != light_index) { return; }
            wi = candidate.sample_direction_pdf.xyz;
            sampled_light_pdf = candidate.sample_direction_pdf.w;
            light_radiance = load_portal_image_spectrum(
                light_index, candidate.position_uv.xy, lambda,
            ) * load_light_scale(light_index);
        } else {
            wi = sample_uniform_infinite_direction(samples.direct.yz);
            sampled_light_pdf = sampled_light_pdf / (4.0 * PI);
            if (light_kind == LIGHT_KIND_IMAGE_INFINITE) {
            light_radiance = load_light_image_spectrum(light_index, wi, lambda) * load_light_scale(light_index);
            } else {
                light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
            }
        }
    } else if (light_kind == LIGHT_KIND_AREA) {
        let total_area = load_area_total(light_payload);
        let distribution_count = load_area_distribution_count(light_payload);
        if (total_area <= 0.0 || distribution_count == 0u) {
            return;
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
            return;
        }
        let b = triangle_sample.xyz;
        light_normal = triangle_geometric_normal(triangle);
        light_position = triangle.p0.xyz * b.x + triangle.p1.xyz * b.y + triangle.p2.xyz * b.z;
        if (!alpha_area_sample_accept(
            light_payload, triangle_selection.primitive, b, light_position,
        )) {
            return;
        }
        light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
        light_error = (abs(triangle.p0.xyz * b.x) + abs(triangle.p1.xyz * b.y)
            + abs(triangle.p2.xyz * b.z)) * gamma(6.0);
        let area_wi = normalize(light_position - light_sample_origin);
        let cosine_light = dot(light_normal, -area_wi);
        if (area_light_is_two_sided(light_payload)) {
            if (abs(cosine_light) == 0.0) {
                return;
            }
        } else if (cosine_light <= 0.0) {
            return;
        }
        sampled_light_pdf = sampled_light_pdf * triangle_selection.pmf * triangle_sample.w;
    } else {
        return;
    }
    if (light_kind != LIGHT_KIND_DISTANT && !is_infinite_light_kind(light_kind)) {
        let to_light = light_position - light_sample_origin;
        distance_squared = dot(to_light, to_light);
        if (distance_squared <= 0.0) {
            return;
        }
        wi = to_light / sqrt(distance_squared);
    }
    if (light_kind == LIGHT_KIND_POINT || light_kind == LIGHT_KIND_SPOT) {
        light_radiance = light_radiance / distance_squared;
    }
    let use_mis = (light_kind == LIGHT_KIND_AREA && !area_light_is_zero_alpha_sample_only(light_payload))
        || is_infinite_light_kind(light_kind);
    direct_light_samples[pixel_index] = DirectLightSample(
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
