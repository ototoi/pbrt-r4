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
    if (ray.depth >= viewport.max_depth || viewport.light_count == 0u) {
        return;
    }
    let light_selection = sample_uniform_light(pixel_index);
    let light_index = light_selection.index;
    let light_kind = load_light_kind(light_index);
    let light_payload = load_light_payload(light_index);
    var light_position = vec3<f32>(0.0);
    var light_radiance = vec3<f32>(0.0);
    var sampled_light_pdf = light_selection.pmf;
    if (light_kind == LIGHT_KIND_POINT) {
        let light = load_point_light(light_payload);
        light_position = light.position.xyz;
        light_radiance = light.intensity.xyz;
    } else if (light_kind == LIGHT_KIND_AREA) {
        let area_instance = instances[load_area_instance(light_payload)];
        let area_geometry = geometries[area_instance.geometry];
        let distribution_offset = load_area_distribution_offset(light_payload);
        let distribution_count = load_area_distribution_count(light_payload);
        let total_area = load_area_total(light_payload);
        if (distribution_count == 0u || total_area <= 0.0) {
            return;
        }
        let selector = random01(pixel_index, 2u);
        var triangle_index = 0u;
        for (var i = 0u; i < distribution_count; i++) {
            if (selector <= load_triangle_cdf(distribution_offset, i)) {
                triangle_index = load_triangle_primitive(distribution_offset, i);
                break;
            }
        }
        let first_index = area_geometry.index_offset + triangle_index * 3u;
        let i0 = area_geometry.vertex_offset + indices[first_index];
        let i1 = area_geometry.vertex_offset + indices[first_index + 1u];
        let i2 = area_geometry.vertex_offset + indices[first_index + 2u];
        let su = sqrt(random01(pixel_index, 3u));
        let bv = random01(pixel_index, 4u);
        let b0 = 1.0 - su;
        let b1 = su * (1.0 - bv);
        let b2 = su * bv;
        let p0 = (area_instance.world_from_object * vertices[i0].position).xyz;
        let p1 = (area_instance.world_from_object * vertices[i1].position).xyz;
        let p2 = (area_instance.world_from_object * vertices[i2].position).xyz;
        light_position = p0 * b0 + p1 * b1 + p2 * b2;
        light_radiance = load_area_emission(light_payload).xyz;
        let light_normal = normalize(cross(p1 - p0, p2 - p0));
        let area_to_light = light_position - surface.position.xyz;
        let area_distance_squared = dot(area_to_light, area_to_light);
        if (area_distance_squared <= 0.0) {
            return;
        }
        let area_wi = area_to_light / sqrt(area_distance_squared);
        let cosine_light = dot(light_normal, -area_wi);
        if (load_area_two_sided(light_payload)) {
            if (abs(cosine_light) == 0.0) {
                return;
            }
        } else if (cosine_light <= 0.0) {
            return;
        }
        sampled_light_pdf = sampled_light_pdf
            * area_distance_squared
            / (max(abs(cosine_light), 1e-7) * total_area);
    } else {
        return;
    }
    let to_light = light_position - surface.position.xyz;
    let distance_squared = dot(to_light, to_light);
    if (distance_squared <= 0.0) {
        return;
    }
    let distance = sqrt(distance_squared);
    let wi = to_light / distance;
    let cosine = max(dot(surface.normal.xyz, wi), 0.0);
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
    let direct = light_radiance * (1.0 / PI) * cosine / sampled_light_pdf * mis_weight;
    append_shadow_ray(
        pixel_index,
        offset_ray_origin(surface.position.xyz, surface.position_error.xyz, surface.normal.xyz),
        wi,
        distance - dot(abs(wi), surface.position_error.xyz),
        direct,
    );
}
