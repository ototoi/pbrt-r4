@compute @workgroup_size(64, 1, 1)
fn shade_surface(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
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
    let object_p0 = vertices[i0].position.xyz;
    let object_p1 = vertices[i1].position.xyz;
    let object_p2 = vertices[i2].position.xyz;
    let p0 = (instance.world_from_object * vertices[i0].position).xyz;
    let p1 = (instance.world_from_object * vertices[i1].position).xyz;
    let p2 = (instance.world_from_object * vertices[i2].position).xyz;
    let b1 = surface.barycentric.x;
    let b2 = surface.barycentric.y;
    let b0 = 1.0 - b1 - b2;
    let position = p0 * b0 + p1 * b1 + p2 * b2;
    let position_error = (abs(p0 * b0) + abs(p1 * b1) + abs(p2 * b2)) * gamma(7.0);
    let uv0 = vertices[i0].uv;
    let uv1 = vertices[i1].uv;
    let uv2 = vertices[i2].uv;
    let uv = uv0 * b0 + uv1 * b1 + uv2 * b2;
    surfaces[pixel_index].position_error = vec4<f32>(position_error, 0.0);
    var geometric_normal = normalize(cross(p1 - p0, p2 - p0));
    let object_normal = vertices[i0].normal.xyz * b0
        + vertices[i1].normal.xyz * b1
        + vertices[i2].normal.xyz * b2;
    let transformed_normal = (instance.normal_from_object * vec4<f32>(object_normal, 0.0)).xyz;
    var normal = geometric_normal;
    if (dot(object_normal, object_normal) > 0.0) {
        normal = normalize(transformed_normal);
    }
    if (instance_orientation_is_reversed(instance.orientation_flags)) {
        geometric_normal = -geometric_normal;
        normal = -normal;
    }
    if (instance_orientation_swaps_handedness(instance.orientation_flags)
        && dot(object_normal, object_normal) > 0.0) {
        normal = -normal;
    }
    let duv02 = uv0 - uv2;
    let duv12 = uv1 - uv2;
    let object_dp02 = object_p0 - object_p2;
    let object_dp12 = object_p1 - object_p2;
    let determinant = fma(duv02.x, duv12.y, -duv02.y * duv12.x);
    var object_dpdu = vec3<f32>(0.0);
    var object_dpdv = vec3<f32>(0.0);
    if (abs(determinant) >= 1e-9) {
        let inverse_determinant = 1.0 / determinant;
        object_dpdu = fma(vec3<f32>(duv12.y), object_dp02, -duv02.y * object_dp12) * inverse_determinant;
        object_dpdv = fma(vec3<f32>(duv02.x), object_dp12, -duv12.x * object_dp02) * inverse_determinant;
    }
    let object_normal_derivative = cross(object_dpdu, object_dpdv);
    if (abs(determinant) < 1e-9 || dot(object_normal_derivative, object_normal_derivative) == 0.0) {
        let object_ng = normalize(cross(object_p2 - object_p0, object_p1 - object_p0));
        object_dpdu = coordinate_system_x(object_ng);
    }
    let object_tangent = vertices[i0].tangent.xyz * b0
        + vertices[i1].tangent.xyz * b1
        + vertices[i2].tangent.xyz * b2;
    var tangent_source = object_dpdu;
    if (dot(object_tangent, object_tangent) > 0.0) {
        tangent_source = object_tangent;
    }
    var tangent = (instance.world_from_object * vec4<f32>(tangent_source, 0.0)).xyz;
    let bitangent = cross(normal, tangent);
    if (dot(bitangent, bitangent) > 0.0) {
        tangent = cross(bitangent, normal);
    } else {
        tangent = coordinate_system_x(normal);
    }
    tangent = normalize(tangent);
    let material_root = material_roots[instance.material_root];
    let material_kind = load_surface_material_kind(material_root);
    surfaces[pixel_index].position = vec4<f32>(position, 1.0);
    surfaces[pixel_index].normal = vec4<f32>(normal, 0.0);
    surfaces[pixel_index].geometric_normal = vec4<f32>(geometric_normal, 0.0);
    surfaces[pixel_index].tangent = vec4<f32>(tangent, 0.0);
    surfaces[pixel_index].uv = uv;
    surfaces[pixel_index].material_root = instance.material_root;
    surfaces[pixel_index].flags = 0u;
    let material_queue_index = append_material_eval(ray_index);
    surfaces[pixel_index].attributes_eval_work_item =
        material_queue_index * material_table.attributes_eval_stride;

    if (material_kind != MATERIAL_KIND_NORMAL && material_kind != MATERIAL_KIND_UV
        && instance.area_light != 0xffffffffu) {
        append_hit_area_light(ray_index);
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
