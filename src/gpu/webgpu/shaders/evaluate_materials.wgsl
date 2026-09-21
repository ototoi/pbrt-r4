fn spot_falloff(light_index: u32, world_direction: vec3<f32>) -> f32 {
    if (light_index >= arrayLength(&light_records)) {
        set_render_error();
        return 0.0;
    }
    let model_index = light_records[light_index].sampling_model;
    if (model_index >= arrayLength(&light_sampling_models)) {
        set_render_error();
        return 0.0;
    }
    let model = light_sampling_models[model_index];
    let local_w = vec3<f32>(
        dot(model.world_to_light0.xyz, world_direction),
        dot(model.world_to_light1.xyz, world_direction),
        dot(model.world_to_light2.xyz, world_direction),
    );
    let local_length_squared = dot(local_w, local_w);
    if (local_length_squared == 0.0) {
        set_render_error();
        return 0.0;
    }
    let cosine = dot(normalize(load_light_direction(light_index)), local_w * inverseSqrt(local_length_squared));
    let falloff_start = load_light_scalar(light_index, 2u);
    let falloff_end = load_light_scalar(light_index, 3u);
    if (falloff_start == falloff_end) {
        return select(0.0, 1.0, cosine >= falloff_start);
    }
    return smoothstep(falloff_end, falloff_start, cosine);
}
fn is_infinite_light_kind(light_kind: u32) -> bool {
    return light_kind == LIGHT_KIND_UNIFORM_INFINITE
        || light_kind == LIGHT_KIND_IMAGE_INFINITE
        || light_kind == LIGHT_KIND_PORTAL_IMAGE_INFINITE;
}

fn sample_uniform_infinite_direction(u: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - 2.0 * min(u.x, 0.99999994);
    let phi = 2.0 * PI * u.y;
    let radial = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(radial * cos(phi), radial * sin(phi), z);
}

@compute @workgroup_size(8, 8, 1)
fn evaluate_materials(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    let surface = surfaces[pixel_index];
    var material_node = 0xffffffffu;
    var material_kind = MATERIAL_KIND_NORMAL;
    let lambda = load_sample_lambda(pixel_index);
    let samples = load_ray_samples(pixel_index);
    let selected_evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);
    material_node = selected_evaluated.material_node;
    material_kind = selected_evaluated.bxdf_kind;
    let root_surface_kind = selected_evaluated.bxdf_kind;
    if (root_surface_kind == MATERIAL_KIND_COATED_DIFFUSE) {
        material_kind = MATERIAL_KIND_DIFFUSE;
    } else if (root_surface_kind == MATERIAL_KIND_COATED_CONDUCTOR) {
        material_kind = MATERIAL_KIND_CONDUCTOR;
    }
    if (surface.hit == 0u
        || (material_kind != MATERIAL_KIND_DIFFUSE
            && material_kind != MATERIAL_KIND_CONDUCTOR
            && material_kind != MATERIAL_KIND_MEASURED)) {
        return;
    }
    var reflectance = vec4<f32>(0.0);
    if (material_kind == MATERIAL_KIND_DIFFUSE) {
        reflectance = selected_evaluated.values[0];
    }
    if (ray.depth >= viewport.max_depth || light_table.light_count == 0u) {
        return;
    }
    let wo = -ray.direction.xyz;
    let light_sample_origin = offset_ray_origin(
        surface.position.xyz,
        surface.position_error.xyz,
        surface.geometric_normal.xyz,
        wo,
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
        light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
        light_error = (abs(triangle.p0.xyz * b.x) + abs(triangle.p1.xyz * b.y)
            + abs(triangle.p2.xyz * b.z)) * gamma(6.0);
        let area_wi = normalize(light_position - light_sample_origin);
        let cosine_light = dot(light_normal, -area_wi);
        if (load_area_two_sided(light_payload)) {
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
    // pbrt-v4 semantics: DiffuseBxDF::f returns R/pi only when wo and wi lie
    // in the same hemisphere of the shading frame (SameHemisphere), and
    // SampleLd weights it with AbsDot(wi, shading.n). This keeps diffuse
    // lighting correct even when the mesh shading normals are globally
    // inverted (e.g. loopsubdiv limit normals wind opposite to the faces).
    let shading_n = surface.normal.xyz;
    let cos_wo = dot(shading_n, wo);
    let cos_wi = dot(shading_n, wi);
    if (cos_wo * cos_wi <= 0.0) {
        return;
    }
    let cosine = abs(cos_wi);
    if (cosine == 0.0) {
        return;
    }
    if (light_kind == LIGHT_KIND_POINT || light_kind == LIGHT_KIND_SPOT) {
        light_radiance = light_radiance / distance_squared;
    }
    var bsdf_pdf = cosine / PI;
    var f = reflectance / PI;
    if (material_kind == MATERIAL_KIND_CONDUCTOR) {
        let eta = selected_evaluated.values[0];
        let k = selected_evaluated.values[1];
        let h = scattering_local(normalize(wo + wi), shading_n);
        let fresnel = conductor_fresnel(dot(scattering_local(wo, shading_n), h), eta, k);
        let alpha = max(selected_evaluated.values[2].x, 1e-3);
        let cos_h = max(abs(h.z), 1e-5);
        let alpha2 = alpha * alpha;
        let d = alpha2 / (PI * pow(cos_h * cos_h * (alpha2 - 1.0) + 1.0, 2.0));
        let cos_o = max(abs(cos_wo), 1e-5);
        let cos_i = max(abs(cos_wi), 1e-5);
        let g_o = 2.0 * cos_o / (cos_o + sqrt(cos_o * cos_o + alpha2 * (1.0 - cos_o * cos_o)));
        let g_i = 2.0 * cos_i / (cos_i + sqrt(cos_i * cos_i + alpha2 * (1.0 - cos_i * cos_i)));
        f = fresnel * d * g_o * g_i / (4.0 * cos_o * cos_i);
        bsdf_pdf = d * cos_h / max(4.0 * abs(dot(scattering_local(wo, shading_n), h)), 1e-5);
    }
    if (material_kind == MATERIAL_KIND_MEASURED) {
        let id = measured_id(material_node);
        if (id == 0xffffffffu) { return; }
        let local_wo = scattering_local_frame(wo, surface.tangent.xyz, shading_n);
        let local_wi = scattering_local_frame(wi, surface.tangent.xyz, shading_n);
        f = measured_f(id, local_wo, local_wi, lambda);
        bsdf_pdf = measured_pdf(id, local_wo, local_wi);
    }
    let surface_kind = selected_evaluated.bxdf_kind;
    if (surface_kind == MATERIAL_KIND_COATED_DIFFUSE || surface_kind == MATERIAL_KIND_COATED_CONDUCTOR) {
        let layered_wo = scattering_local(wo, shading_n);
        let layered_wi = scattering_local(wi, shading_n);
        f = evaluate_layered_f(
            selected_evaluated,
            surface_kind,
            layered_wo,
            layered_wi,
            pixel_index,
            ray.depth,
        );
        bsdf_pdf = evaluate_layered_pdf(
            selected_evaluated,
            surface_kind,
            layered_wo,
            layered_wi,
            pixel_index,
            ray.depth,
        );
    }
    var mis_weight = 1.0;
    if (light_kind == LIGHT_KIND_AREA || is_infinite_light_kind(light_kind)) {
        let light_pdf2 = sampled_light_pdf * sampled_light_pdf;
        let bsdf_pdf2 = bsdf_pdf * bsdf_pdf;
        mis_weight = light_pdf2 / max(light_pdf2 + bsdf_pdf2, 1e-7);
    }
    let direct = light_radiance * f * cosine
        / (max(ray.inv_w_u, 1e-7) * sampled_light_pdf)
        * mis_weight;
    let shadow_origin = light_sample_origin;
    var shadow_direction = wi;
    var shadow_distance = RAY_T_MAX;
    if (light_kind == LIGHT_KIND_AREA) {
        let shadow_target = offset_ray_origin(light_position, light_error, light_normal, -wi);
        let shadow_vector = shadow_target - shadow_origin;
        shadow_distance = length(shadow_vector);
        shadow_direction = shadow_vector / shadow_distance;
    } else if (light_kind != LIGHT_KIND_DISTANT && !is_infinite_light_kind(light_kind)) {
        let shadow_vector = light_position - shadow_origin;
        shadow_distance = length(shadow_vector);
        shadow_direction = shadow_vector / shadow_distance;
    }
    append_shadow_ray(
        pixel_index,
        shadow_origin,
        shadow_direction,
        shadow_distance,
        (ray.throughput * direct),
    );
}
