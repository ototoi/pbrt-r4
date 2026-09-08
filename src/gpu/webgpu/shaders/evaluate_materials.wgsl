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
    let material_index = resolve_material_leaf(surface.material);
    let material_kind = load_material_kind(material_index);
    if (surface.hit == 0u
        || (material_kind != MATERIAL_KIND_DIFFUSE
            && material_kind != MATERIAL_KIND_CONDUCTOR)) {
        return;
    }
    let lambda = load_sample_lambda(pixel_index);
    var reflectance = vec4<f32>(0.0);
    if (material_kind == MATERIAL_KIND_DIFFUSE) {
        reflectance = load_diffuse_reflectance(material_index, lambda);
    }
    let samples = load_ray_samples(pixel_index);
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
    if (light_kind == LIGHT_KIND_POINT) {
        light_position = load_point_position(light_index);
        light_radiance = load_light_spectrum(light_index, 0u, lambda) * load_light_scale(light_index);
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
            vec2<f32>(samples.direct.z, samples.direct.w),
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
    let to_light = light_position - light_sample_origin;
    let distance_squared = dot(to_light, to_light);
    if (distance_squared <= 0.0) {
        return;
    }
    let distance = sqrt(distance_squared);
    let wi = to_light / distance;
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
    if (light_kind == LIGHT_KIND_POINT) {
        light_radiance = light_radiance / distance_squared;
    }
    var bsdf_pdf = cosine / PI;
    var f = reflectance / PI;
    if (material_kind == MATERIAL_KIND_CONDUCTOR) {
        let eta = load_conductor_eta(material_index, lambda);
        let k = load_conductor_k(material_index, lambda);
        let h = scattering_local(normalize(wo + wi), shading_n);
        let fresnel = conductor_fresnel(dot(scattering_local(wo, shading_n), h), eta, k);
        let alpha = max(load_conductor_roughness(material_index), 1e-3);
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
    var mis_weight = 1.0;
    if (light_kind == LIGHT_KIND_AREA) {
        mis_weight = sampled_light_pdf / max(sampled_light_pdf + bsdf_pdf, 1e-7);
    }
    let direct = light_radiance * f * cosine
        / (max(ray.inv_w_u, 1e-7) * sampled_light_pdf)
        * mis_weight;
    let shadow_origin = light_sample_origin;
    var shadow_target = light_position;
    if (light_kind == LIGHT_KIND_AREA) {
        shadow_target = offset_ray_origin(light_position, light_error, light_normal, -wi);
    }
    let shadow_vector = shadow_target - shadow_origin;
    let shadow_distance = length(shadow_vector);
    if (shadow_distance <= 0.0) {
        return;
    }
    append_shadow_ray(
        pixel_index,
        shadow_origin,
        shadow_vector / shadow_distance,
        shadow_distance,
        (ray.throughput * direct),
    );
}
