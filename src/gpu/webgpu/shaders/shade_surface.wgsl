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
    if (surface.hit == 0u) {
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
    surfaces[pixel_index].position_error = vec4<f32>(abs(position) * gamma(3.0), 0.0);
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
    surfaces[pixel_index].geometric_normal = vec4<f32>(geometric_normal, 0.0);
    surfaces[pixel_index].material = instance.material;
    surfaces[pixel_index].flags = 0u;
    append_material_eval(pixel_index);

    if (material_kind == MATERIAL_KIND_DIFFUSE && instance.area_light != 0xffffffffu) {
        append_hit_area_light(pixel_index);
    }

    if (material_kind == MATERIAL_KIND_NORMAL) {
        store_sample_radiance(pixel_index, vec4<f32>(geometric_normal * 0.5 + vec3<f32>(0.5), 1.0));
        surfaces[pixel_index].flags = 1u;
    } else if (material_kind == MATERIAL_KIND_UV) {
        let uv = vertices[i0].uv * b0 + vertices[i1].uv * b1 + vertices[i2].uv * b2;
        store_sample_radiance(pixel_index, vec4<f32>(uv.x, uv.y, 0.0, 1.0));
        surfaces[pixel_index].flags = 1u;
    }
}
