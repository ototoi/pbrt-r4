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
    let light_sample = direct_light_samples[pixel_index];
    if (light_sample.valid == 0u) {
        return;
    }
    let surface = surfaces[pixel_index];
    let lambda = load_sample_lambda(pixel_index);
    let selected_evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);
    let material_node = selected_evaluated.material_node;
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
    var reflectance = vec4<f32>(0.0);
    var transmittance = vec4<f32>(0.0);
    if (material_kind == MATERIAL_KIND_DIFFUSE) {
        reflectance = selected_evaluated.values[0];
    } else if (material_kind == MATERIAL_KIND_DIFFUSE_TRANSMISSION) {
        reflectance = selected_evaluated.values[0];
        transmittance = selected_evaluated.values[1];
    }
    let wo = -ray.direction.xyz;
    let wi = light_sample.direction_pdf.xyz;
    let sampled_light_pdf = light_sample.direction_pdf.w;
    let light_radiance = light_sample.radiance;
    let light_kind = light_sample.light_kind;
    let light_position = light_sample.position;
    let light_error = light_sample.position_error;
    let light_normal = light_sample.normal;
    // pbrt-v4 semantics: DiffuseBxDF::f returns R/pi only when wo and wi lie
    // in the same hemisphere of the shading frame (SameHemisphere), and
    // SampleLd weights it with AbsDot(wi, shading.n). This keeps diffuse
    // lighting correct even when the mesh shading normals are globally
    // inverted (e.g. loopsubdiv limit normals wind opposite to the faces).
    let shading_n = surface.normal.xyz;
    let cos_wo = dot(shading_n, wo);
    let cos_wi = dot(shading_n, wi);
    if (material_kind != MATERIAL_KIND_DIFFUSE_TRANSMISSION && cos_wo * cos_wi <= 0.0) {
        return;
    }
    let cosine = abs(cos_wi);
    if (cosine == 0.0) {
        return;
    }
    var bsdf_pdf = cosine / PI;
    var f = reflectance / PI;
    if (material_kind == MATERIAL_KIND_DIFFUSE_TRANSMISSION) {
        let pr = max_spectrum(reflectance);
        let pt = max_spectrum(transmittance);
        let total = pr + pt;
        if (total <= 0.0) { return; }
        let same_side = cos_wo * cos_wi > 0.0;
        let branch_probability = select(pt / total, pr / total, same_side);
        f = select(transmittance, reflectance, same_side) / PI;
        bsdf_pdf = cosine / PI * branch_probability;
    } else if (material_kind == MATERIAL_KIND_CONDUCTOR_ETA_K || material_kind == MATERIAL_KIND_CONDUCTOR_REFLECTANCE) {
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
    if (light_sample.use_mis != 0u) {
        let light_pdf2 = sampled_light_pdf * sampled_light_pdf;
        let bsdf_pdf2 = bsdf_pdf * bsdf_pdf;
        mis_weight = light_pdf2 / max(light_pdf2 + bsdf_pdf2, 1e-7);
    }
    let direct = light_radiance * f * cosine
        / (max(ray.inv_w_u, 1e-7) * sampled_light_pdf)
        * mis_weight;
    // Spawn the shadow ray from the side containing the sampled light.
    let shadow_origin = offset_ray_origin(
        surface.position.xyz,
        surface.position_error.xyz,
        surface.geometric_normal.xyz,
        wi,
    );
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
