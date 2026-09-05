@compute @workgroup_size(8, 8, 1)
fn evaluate_materials(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let queue_index = global_id.y * viewport.width + global_id.x;
    if (queue_index >= material_eval_count()) {
        return;
    }
    let pixel_index = load_material_eval_pixel(queue_index);
    let surface = surfaces[pixel_index];
    if (surface.hit == 0u || load_material_kind(surface.material) != MATERIAL_KIND_DIFFUSE) {
        return;
    }
    let ray_index = find_current_ray_for_pixel(pixel_index);
    if (ray_index == 0xffffffffu) {
        return;
    }
    let ray = load_current_ray(ray_index);
    let samples = load_ray_samples(pixel_index);
    if (ray.depth >= viewport.max_depth || viewport.light_count == 0u) {
        return;
    }
    let wo = -ray.direction.xyz;
    let light_sample_origin = offset_ray_origin(
        surface.position.xyz,
        surface.position_error.xyz,
        surface.geometric_normal.xyz,
        wo,
    );
    let light_selection = sample_uniform_light(samples.direct.x);
    let light_index = light_selection.index;
    let light_kind = load_light_kind(light_index);
    let light_payload = load_light_payload(light_index);
    var light_position = vec3<f32>(0.0);
    var light_error = vec3<f32>(0.0);
    var light_normal = vec3<f32>(0.0);
    var light_radiance = vec3<f32>(0.0);
    var sampled_light_pdf = light_selection.pmf;
    if (light_kind == LIGHT_KIND_POINT) {
        let light = load_point_light(light_payload);
        light_position = light.position.xyz;
        light_radiance = light.intensity.xyz;
    } else if (light_kind == LIGHT_KIND_AREA) {
        let total_area = load_area_total(light_payload);
        if (total_area <= 0.0) {
            return;
        }
        let triangle = load_area_triangle(light_payload);
        let triangle_sample = sample_triangle_for_context(
            triangle,
            light_sample_origin,
            surface.normal.xyz,
            total_area,
            vec2<f32>(samples.direct.y, samples.direct.z),
        );
        if (triangle_sample.w <= 0.0) {
            return;
        }
        let b = triangle_sample.xyz;
        light_normal = triangle_sample_normal(triangle, b);
        light_position = triangle.p0.xyz * b.x + triangle.p1.xyz * b.y + triangle.p2.xyz * b.z;
        light_radiance = load_area_emission(light_payload).xyz;
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
        sampled_light_pdf = sampled_light_pdf * triangle_sample.w;
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
    let bsdf_pdf = cosine / PI;
    var mis_weight = 1.0;
    if (light_kind == LIGHT_KIND_AREA) {
        mis_weight = sampled_light_pdf / max(sampled_light_pdf + bsdf_pdf, 1e-7);
    }
    let direct = light_radiance * (1.0 / PI) * cosine
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
        direct,
    );
}
