@compute @workgroup_size(8, 8, 1)
fn shade_surface(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= current_ray_count()) {
        return;
    }
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    if (ray.is_active == 0u || surface.hit == 0u) {
        if (ray.is_active != 0u) {
            var inactive_ray = ray;
            inactive_ray.is_active = 0u;
            store_current_ray(ray_index, inactive_ray);
        }
        return;
    }

    let instance = instances[surface.instance_custom_data];
    let geometry = geometries[instance.geometry];
    let first_index = geometry.index_offset + surface.primitive_index * 3u;
    let i0 = geometry.vertex_offset + indices[first_index];
    let i1 = geometry.vertex_offset + indices[first_index + 1u];
    let i2 = geometry.vertex_offset + indices[first_index + 2u];
    let p0 = (instance.world_from_object * vertices[i0].position).xyz;
    let p1 = (instance.world_from_object * vertices[i1].position).xyz;
    let p2 = (instance.world_from_object * vertices[i2].position).xyz;
    let b1 = surface.barycentric.x;
    let b2 = surface.barycentric.y;
    let b0 = 1.0 - b1 - b2;
    let position = ray.origin.xyz + ray.direction.xyz * surface.t;
    let geometric_normal = normalize(cross(p1 - p0, p2 - p0));
    let object_normal = vertices[i0].normal.xyz * b0
        + vertices[i1].normal.xyz * b1
        + vertices[i2].normal.xyz * b2;
    let transformed_normal = (instance.normal_from_object * vec4<f32>(object_normal, 0.0)).xyz;
    var normal = geometric_normal;
    if (dot(object_normal, object_normal) > 0.0) {
        normal = normalize(transformed_normal);
    }
    let material_kind = load_material_kind(instance.material);
    surfaces[pixel_index].position = vec4<f32>(position, 1.0);
    surfaces[pixel_index].normal = vec4<f32>(normal, 0.0);
    surfaces[pixel_index].material = instance.material;
    surfaces[pixel_index].flags = 0u;
    surfaces[pixel_index].direct = vec4<f32>(0.0);
    surfaces[pixel_index].shadow_visible = 0u;

    if (material_kind == MATERIAL_KIND_DIFFUSE && instance.area_light != 0xffffffffu) {
        framebuffer[pixel_index] = framebuffer[pixel_index]
            + ray.throughput * load_area_emission(instance.area_light);
    }

    if (material_kind == MATERIAL_KIND_NORMAL) {
        framebuffer[pixel_index] = vec4<f32>(geometric_normal * 0.5 + vec3<f32>(0.5), 1.0);
        surfaces[pixel_index].flags = 1u;
        var inactive_ray = ray;
        inactive_ray.is_active = 0u;
        store_current_ray(ray_index, inactive_ray);
        return;
    }
    if (material_kind == MATERIAL_KIND_UV) {
        let uv = vertices[i0].uv * b0 + vertices[i1].uv * b1 + vertices[i2].uv * b2;
        framebuffer[pixel_index] = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
        surfaces[pixel_index].flags = 1u;
        var inactive_ray = ray;
        inactive_ray.is_active = 0u;
        store_current_ray(ray_index, inactive_ray);
        return;
    }
    if (material_kind != MATERIAL_KIND_DIFFUSE
        || ray.depth >= viewport.max_depth
        || viewport.light_count == 0u) {
        return;
    }
    let light_index = hash_u32(viewport.seed ^ pixel_index ^ viewport.sample_index) % viewport.light_count;
    let light_kind = load_light_kind(light_index);
    let light_payload = load_light_payload(light_index);
    var light_position = vec3<f32>(0.0);
    var light_radiance = vec3<f32>(0.0);
    var sampled_light_pdf = 1.0 / f32(viewport.light_count);
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
        let area_to_light = light_position - position;
        let area_distance_squared = dot(area_to_light, area_to_light);
        if (area_distance_squared <= 0.0) {
            return;
        }
        let area_distance = sqrt(area_distance_squared);
        let area_wi = area_to_light / area_distance;
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
    let to_light = light_position - position;
    let distance_squared = dot(to_light, to_light);
    if (distance_squared <= 0.0) {
        return;
    }
    let distance = sqrt(distance_squared);
    let wi = to_light / distance;
    let cosine = max(dot(normal, wi), 0.0);
    if (cosine == 0.0) {
        return;
    }
    if (light_kind == LIGHT_KIND_POINT) {
        light_radiance = light_radiance / distance_squared;
    }
    let direct = light_radiance * (0.5 / PI) * cosine / sampled_light_pdf;
    surfaces[pixel_index].shadow_origin = vec4<f32>(position + normal * RAY_EPSILON, 1.0);
    surfaces[pixel_index].shadow_direction = vec4<f32>(wi, 0.0);
    surfaces[pixel_index].shadow_t = distance - RAY_EPSILON;
    surfaces[pixel_index].direct = vec4<f32>(direct, 0.0);
    append_shadow_ray(pixel_index);
}
