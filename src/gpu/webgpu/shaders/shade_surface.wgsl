fn rotate_from_to(from_direction: vec3<f32>, to_direction: vec3<f32>, value: vec3<f32>) -> vec3<f32> {
    var reflection = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(from_direction.x) < 0.72 && abs(to_direction.x) < 0.72) {
        reflection = vec3<f32>(1.0, 0.0, 0.0);
    } else if (abs(from_direction.y) < 0.72 && abs(to_direction.y) < 0.72) {
        reflection = vec3<f32>(0.0, 1.0, 0.0);
    }
    let u = reflection - from_direction;
    let v = reflection - to_direction;
    let uu = dot(u, u);
    let vv = dot(v, v);
    return value - 2.0 * u * (dot(u, value) / uu)
        - 2.0 * v * (dot(v, value) / vv)
        + 4.0 * dot(u, v) * v * dot(u, value) / (uu * vv);
}

fn camera_approximate_dp_dxy(position: vec3<f32>, normal: vec3<f32>) -> mat2x3<f32> {
    if (camera.disable_texture_filtering != 0u) {
        return mat2x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let camera_position = (camera.world_to_camera * vec4<f32>(position, 1.0)).xyz;
    let camera_normal = normalize((transpose(camera.camera_to_world) * vec4<f32>(normal, 0.0)).xyz);
    let camera_axis = normalize(camera_position);
    let down_normal = rotate_from_to(camera_axis, vec3<f32>(0.0, 0.0, 1.0), camera_normal);
    let down_position = rotate_from_to(camera_axis, vec3<f32>(0.0, 0.0, 1.0), camera_position);
    let plane_d = down_normal.z * down_position.z;
    let x_direction = vec3<f32>(0.0, 0.0, 1.0) + camera.min_dir_differential_x.xyz;
    let y_direction = vec3<f32>(0.0, 0.0, 1.0) + camera.min_dir_differential_y.xyz;
    let tx = plane_d / dot(down_normal, x_direction);
    let ty = plane_d / dot(down_normal, y_direction);
    let px = tx * x_direction;
    let py = ty * y_direction;
    var scale = max(0.125, inverseSqrt(f32(sampler_params.samples_per_pixel)));
    if (camera.disable_pixel_jitter != 0u) {
        scale = 1.0;
    }
    let dx_camera = scale * rotate_from_to(vec3<f32>(0.0, 0.0, 1.0), camera_axis, px - down_position);
    let dy_camera = scale * rotate_from_to(vec3<f32>(0.0, 0.0, 1.0), camera_axis, py - down_position);
    let dx_world = (camera.camera_to_world * vec4<f32>(dx_camera, 0.0)).xyz;
    let dy_world = (camera.camera_to_world * vec4<f32>(dy_camera, 0.0)).xyz;
    return mat2x3<f32>(dx_world, dy_world);
}

fn surface_uv_differentials(
    dpdu: vec3<f32>, dpdv: vec3<f32>, dpdx: vec3<f32>, dpdy: vec3<f32>,
) -> vec4<f32> {
    let ata00 = dot(dpdu, dpdu);
    let ata01 = dot(dpdu, dpdv);
    let ata11 = dot(dpdv, dpdv);
    let determinant = fma(ata00, ata11, -ata01 * ata01);
    var inverse_determinant = 0.0;
    if (abs(determinant) > 0.0 && abs(determinant) < 3.402823466e+38) {
        inverse_determinant = 1.0 / determinant;
    }
    let atb0x = dot(dpdu, dpdx);
    let atb1x = dot(dpdv, dpdx);
    let atb0y = dot(dpdu, dpdy);
    let atb1y = dot(dpdv, dpdy);
    var dudx = fma(ata11, atb0x, -ata01 * atb1x) * inverse_determinant;
    var dvdx = fma(ata00, atb1x, -ata01 * atb0x) * inverse_determinant;
    var dudy = fma(ata11, atb0y, -ata01 * atb1y) * inverse_determinant;
    var dvdy = fma(ata00, atb1y, -ata01 * atb0y) * inverse_determinant;
    dudx = select(0.0, clamp(dudx, -1e8, 1e8), abs(dudx) < 3.402823466e+38);
    dvdx = select(0.0, clamp(dvdx, -1e8, 1e8), abs(dvdx) < 3.402823466e+38);
    dudy = select(0.0, clamp(dudy, -1e8, 1e8), abs(dudy) < 3.402823466e+38);
    dvdy = select(0.0, clamp(dvdy, -1e8, 1e8), abs(dvdy) < 3.402823466e+38);
    return vec4<f32>(dudx, dvdx, dudy, dvdy);
}

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
    let position_error = (abs(p0 * b0) + abs(p1 * b1) + abs(p2 * b2)) * gamma(7.0)
        + hardware_intersection_error(p0, p1, p2);
    let uv0 = vertices[i0].uv;
    let uv1 = vertices[i1].uv;
    let uv2 = vertices[i2].uv;
    let uv = uv0 * b0 + uv1 * b1 + uv2 * b2;
    surfaces[pixel_index].position_error = vec4<f32>(position_error, 0.0);
    var geometric_normal = normalize(cross(p1 - p0, p2 - p0));
    if (intersection_flips_geometric_normal(geometry.intersection_normal_kind, instance.orientation_flags)) {
        geometric_normal = -geometric_normal;
    }
    let object_normal = vertices[i0].normal.xyz * b0
        + vertices[i1].normal.xyz * b1
        + vertices[i2].normal.xyz * b2;
    var normal = triangle_shading_normal(instance, geometry, object_normal, geometric_normal);
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
        object_dpdv = coordinate_system_y(object_ng);
    }
    var object_dndu = vec3<f32>(0.0);
    var object_dndv = vec3<f32>(0.0);
    if (dot(object_normal, object_normal) > 0.0) {
        let dn1 = vertices[i0].normal.xyz - vertices[i2].normal.xyz;
        let dn2 = vertices[i1].normal.xyz - vertices[i2].normal.xyz;
        if (abs(determinant) < 1e-9) {
            let dn = cross(vertices[i2].normal.xyz - vertices[i0].normal.xyz,
                vertices[i1].normal.xyz - vertices[i0].normal.xyz);
            if (dot(dn, dn) > 0.0) {
                object_dndu = coordinate_system_x(dn);
                object_dndv = coordinate_system_y(dn);
            }
        } else {
            let inverse_determinant = 1.0 / determinant;
            object_dndu = fma(vec3<f32>(duv12.y), dn1, -duv02.y * dn2) * inverse_determinant;
            object_dndv = fma(vec3<f32>(duv02.x), dn2, -duv12.x * dn1) * inverse_determinant;
        }
    }
    var dpdu = (instance.world_from_object * vec4<f32>(object_dpdu, 0.0)).xyz;
    var dpdv = (instance.world_from_object * vec4<f32>(object_dpdv, 0.0)).xyz;
    var dndu = (instance.normal_from_object * vec4<f32>(object_dndu, 0.0)).xyz;
    var dndv = (instance.normal_from_object * vec4<f32>(object_dndv, 0.0)).xyz;
    // v4 SurfaceInteraction flips quadric normals, but not their derivatives.
    // Undo ReverseOrientation already applied to the flattened vertex normals.
    if (geometry.intersection_normal_kind == INTERSECTION_NORMAL_KIND_QUADRIC
        && instance_orientation_is_reversed(instance.orientation_flags)) {
        dndu = -dndu;
        dndv = -dndv;
    }
    let dp_dxy = camera_approximate_dp_dxy(position, geometric_normal);
    let uv_differentials = surface_uv_differentials(dpdu, dpdv, dp_dxy[0], dp_dxy[1]);
    let object_tangent = vertices[i0].tangent.xyz * b0
        + vertices[i1].tangent.xyz * b1
        + vertices[i2].tangent.xyz * b2;
    var tangent_source = object_dpdu;
    if (dot(object_tangent, object_tangent) > 0.0) {
        tangent_source = object_tangent;
    }
    var tangent = (instance.world_from_object * vec4<f32>(tangent_source, 0.0)).xyz;
    let shading_bitangent = cross(normal, tangent);
    if (dot(shading_bitangent, shading_bitangent) > 0.0) {
        tangent = cross(shading_bitangent, normal);
    } else {
        tangent = coordinate_system_x(normal);
    }
    tangent = normalize(tangent);
    // pbrt-v4 Triangle::InteractionFromIntersection: meshes with normals or
    // tangents use shading dpdu/dpdv built around the shading normal. Bump
    // mapping reads these; the geometric ones above only feed differentials.
    var shading_dpdu = dpdu;
    var shading_dpdv = dpdv;
    if (dot(object_normal, object_normal) > 0.0 || dot(object_tangent, object_tangent) > 0.0) {
        var ss = dpdu;
        if (dot(object_tangent, object_tangent) > 0.0) {
            ss = (instance.world_from_object * vec4<f32>(object_tangent, 0.0)).xyz;
        }
        let ts = cross(normal, ss);
        if (dot(ts, ts) > 0.0) {
            shading_dpdu = cross(ts, normal);
            shading_dpdv = ts;
        } else {
            shading_dpdu = coordinate_system_x(normal);
            shading_dpdv = coordinate_system_y(normal);
        }
    }
    let material_root = material_roots[instance.material_root];
    let material_kind = load_surface_material_kind(material_root);
    surfaces[pixel_index].position = vec4<f32>(position, 1.0);
    surfaces[pixel_index].normal = vec4<f32>(normal, 0.0);
    surfaces[pixel_index].geometric_normal = vec4<f32>(geometric_normal, 0.0);
    surfaces[pixel_index].tangent = vec4<f32>(tangent, 0.0);
    surfaces[pixel_index].dpdu = vec4<f32>(shading_dpdu, 0.0);
    surfaces[pixel_index].dpdv = vec4<f32>(shading_dpdv, 0.0);
    surfaces[pixel_index].dndu = vec4<f32>(dndu, 0.0);
    surfaces[pixel_index].dndv = vec4<f32>(dndv, 0.0);
    surfaces[pixel_index].dpdx = vec4<f32>(dp_dxy[0], 0.0);
    surfaces[pixel_index].dpdy = vec4<f32>(dp_dxy[1], 0.0);
    surfaces[pixel_index].uv = uv;
    surfaces[pixel_index].uv_differentials = uv_differentials;
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
