fn shadow_visible(
    origin: vec3<f32>,
    direction: vec3<f32>,
    source_primitive: u32,
    source_triangle: u32,
) -> bool {
    var query: ray_query;
    rayQueryInitialize(
        &query,
        acceleration,
        RayDesc(0u, 0xFFu, 0.0, 0.9999, origin, direction),
    );
    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        if (candidate.instance_custom_data == source_primitive
            && candidate.primitive_index == source_triangle) {
            continue;
        }
        let primitive = primitives[candidate.instance_custom_data];
        let index_offset = primitive.first_index + candidate.primitive_index * 3u;
        let i0 = primitive.first_vertex + indices[index_offset];
        let i1 = primitive.first_vertex + indices[index_offset + 1u];
        let i2 = primitive.first_vertex + indices[index_offset + 2u];
        let barycentrics = vec3<f32>(
            1.0 - candidate.barycentrics.x - candidate.barycentrics.y,
            candidate.barycentrics.x,
            candidate.barycentrics.y,
        );
        let uv = vertices[i0].uv.xy * barycentrics.x
            + vertices[i1].uv.xy * barycentrics.y
            + vertices[i2].uv.xy * barycentrics.z;
        if (primitive.alpha == 0xffffffffu
            || alpha_accept(
                sample_float_texture(primitive.alpha, uv, vec4<f32>(0.0)),
                origin,
                direction,
            )) {
            rayQueryConfirmIntersection(&query);
        }
    }
    return rayQueryGetCommittedIntersection(&query).kind == RAY_QUERY_INTERSECTION_NONE;
}

fn render_sample(
    pixel: vec2<f32>,
    lens_sample: vec2<f32>,
    sample_pixel: vec2<i32>,
    sample_index: u32,
) -> vec3<f32> {
    let camera_target = transform(
        camera.camera_from_raster,
        vec4<f32>(pixel, 0.0, 1.0),
    );
    var camera_origin = vec3<f32>(0.0);
    var camera_direction = normalize(camera_target.xyz);
    if (camera.camera_info.x > 0.0) {
        let lens = camera.camera_info.x * sample_uniform_disk_concentric(lens_sample);
        let focus_t = camera.camera_info.y / camera_direction.z;
        let focus = focus_t * camera_direction;
        camera_origin = vec3<f32>(lens, 0.0);
        camera_direction = normalize(focus - camera_origin);
    }
    var ray_origin = transform(camera.render_from_camera, vec4<f32>(camera_origin, 1.0)).xyz;
    var ray_direction = normalize(transform(camera.render_from_camera, vec4<f32>(camera_direction, 0.0)).xyz);
    let x_target = transform(
        camera.camera_from_raster,
        vec4<f32>(pixel + vec2<f32>(1.0, 0.0), 0.0, 1.0),
    );
    let y_target = transform(
        camera.camera_from_raster,
        vec4<f32>(pixel + vec2<f32>(0.0, 1.0), 0.0, 1.0),
    );
    var rx_camera_origin = vec3<f32>(0.0);
    var ry_camera_origin = vec3<f32>(0.0);
    var rx_camera_direction = normalize(x_target.xyz);
    var ry_camera_direction = normalize(y_target.xyz);
    if (camera.camera_info.x > 0.0) {
        let lens = camera.camera_info.x * sample_uniform_disk_concentric(lens_sample);
        rx_camera_origin = vec3<f32>(lens, 0.0);
        ry_camera_origin = rx_camera_origin;
        let rx_focus = (camera.camera_info.y / rx_camera_direction.z) * rx_camera_direction;
        let ry_focus = (camera.camera_info.y / ry_camera_direction.z) * ry_camera_direction;
        rx_camera_direction = normalize(rx_focus - rx_camera_origin);
        ry_camera_direction = normalize(ry_focus - ry_camera_origin);
    }
    var ray_rx_origin = transform(camera.render_from_camera, vec4<f32>(rx_camera_origin, 1.0)).xyz;
    var ray_ry_origin = transform(camera.render_from_camera, vec4<f32>(ry_camera_origin, 1.0)).xyz;
    var ray_rx_direction = normalize(transform(camera.render_from_camera, vec4<f32>(rx_camera_direction, 0.0)).xyz);
    var ray_ry_direction = normalize(transform(camera.render_from_camera, vec4<f32>(ry_camera_direction, 0.0)).xyz);

    var color = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var previous_context_position = vec3<f32>(0.0);
    var previous_context_shading_normal = vec3<f32>(0.0);
    var previous_bsdf_pdf = 0.0;
    for (var depth = 0u; depth <= camera.bvh_info.z; depth += 1u) {
    let origin = ray_origin;
    let direction = ray_direction;
    let rx_origin = ray_rx_origin;
    let ry_origin = ray_ry_origin;
    let rx_direction = ray_rx_direction;
    let ry_direction = ray_ry_direction;
    var query: ray_query;
    rayQueryInitialize(
        &query,
        acceleration,
        RayDesc(0u, 0xFFu, 0.0001, 1.0e30, origin, direction),
    );
    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        let primitive = primitives[candidate.instance_custom_data];
        let index_offset = primitive.first_index + candidate.primitive_index * 3u;
        let i0 = primitive.first_vertex + indices[index_offset];
        let i1 = primitive.first_vertex + indices[index_offset + 1u];
        let i2 = primitive.first_vertex + indices[index_offset + 2u];
        let barycentrics = vec3<f32>(
            1.0 - candidate.barycentrics.x - candidate.barycentrics.y,
            candidate.barycentrics.x,
            candidate.barycentrics.y,
        );
        let uv = vertices[i0].uv.xy * barycentrics.x
            + vertices[i1].uv.xy * barycentrics.y
            + vertices[i2].uv.xy * barycentrics.z;
        if (primitive.alpha == 0xffffffffu
            || alpha_accept(
                sample_float_texture(primitive.alpha, uv, vec4<f32>(0.0)),
                origin,
                direction,
            )) {
            rayQueryConfirmIntersection(&query);
        }
    }
    let intersection = rayQueryGetCommittedIntersection(&query);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        let primitive = primitives[intersection.instance_custom_data];
        let index_offset = primitive.first_index + intersection.primitive_index * 3u;
        let i0 = primitive.first_vertex + indices[index_offset];
        let i1 = primitive.first_vertex + indices[index_offset + 1u];
        let i2 = primitive.first_vertex + indices[index_offset + 2u];
        let barycentrics = vec3<f32>(
            1.0 - intersection.barycentrics.x - intersection.barycentrics.y,
            intersection.barycentrics.x,
            intersection.barycentrics.y,
        );
        let object_position = vertices[i0].position.xyz * barycentrics.x
            + vertices[i1].position.xyz * barycentrics.y
            + vertices[i2].position.xyz * barycentrics.z;
        let geometric_normal = normalize(cross(
            vertices[i1].position.xyz - vertices[i0].position.xyz,
            vertices[i2].position.xyz - vertices[i0].position.xyz,
        ));
        let transform_table = transforms[intersection.instance_custom_data];
        let position = transform(transform_table.render_from_object, vec4<f32>(object_position, 1.0)).xyz;
        let object_position_error = 4.172327e-7 * (
            abs(barycentrics.x * vertices[i0].position.xyz)
            + abs(barycentrics.y * vertices[i1].position.xyz)
            + abs(barycentrics.z * vertices[i2].position.xyz)
        );
        let position_error = transformed_position_error(
            transform_table.render_from_object,
            object_position,
            object_position_error,
        );
        let render_p0 = transform(transform_table.render_from_object, vertices[i0].position).xyz;
        let render_p1 = transform(transform_table.render_from_object, vertices[i1].position).xyz;
        let render_p2 = transform(transform_table.render_from_object, vertices[i2].position).xyz;
        var geometric_render_normal = normalize(cross(render_p1 - render_p0, render_p2 - render_p0));
        if ((primitive.flags & 1u) != 0u) {
            geometric_render_normal = -geometric_render_normal;
        }
        let material = materials[primitive.material];
        let interpolated_normal = vertices[i0].normal.xyz * barycentrics.x
            + vertices[i1].normal.xyz * barycentrics.y
            + vertices[i2].normal.xyz * barycentrics.z;
        let orientation_sign = select(1.0, -1.0, (primitive.flags & 1u) != 0u);
        let has_vertex_normal = dot(interpolated_normal, interpolated_normal) > 1.0e-20;
        var object_normal = geometric_normal;
        if (has_vertex_normal) {
            object_normal = normalize(interpolated_normal);
        }
        object_normal *= orientation_sign;
        let uv = vertices[i0].uv.xy * barycentrics.x
            + vertices[i1].uv.xy * barycentrics.y
            + vertices[i2].uv.xy * barycentrics.z;
        let object_dpdu = triangle_dpdu(
            vertices[i0].position.xyz,
            vertices[i1].position.xyz,
            vertices[i2].position.xyz,
            vertices[i0].uv.xy,
            vertices[i1].uv.xy,
            vertices[i2].uv.xy,
            object_normal,
        );
        let object_dpdv = triangle_dpdv(
            vertices[i0].position.xyz,
            vertices[i1].position.xyz,
            vertices[i2].position.xyz,
            vertices[i0].uv.xy,
            vertices[i1].uv.xy,
            vertices[i2].uv.xy,
            object_normal,
        );
        let object_dndu = orientation_sign * normal_derivative_u(
            vertices[i0].normal.xyz,
            vertices[i1].normal.xyz,
            vertices[i2].normal.xyz,
            vertices[i0].uv.xy,
            vertices[i1].uv.xy,
            vertices[i2].uv.xy,
        );
        let object_dndv = orientation_sign * normal_derivative_v(
            vertices[i0].normal.xyz,
            vertices[i1].normal.xyz,
            vertices[i2].normal.xyz,
            vertices[i0].uv.xy,
            vertices[i1].uv.xy,
            vertices[i2].uv.xy,
        );
        let object_tangent = vertices[i0].tangent.xyz * barycentrics.x
            + vertices[i1].tangent.xyz * barycentrics.y
            + vertices[i2].tangent.xyz * barycentrics.z;
        var normal = geometric_render_normal;
        if (has_vertex_normal) {
            normal = normalize(transform(transform_table.normal_from_object, vec4<f32>(object_normal, 0.0)).xyz);
            geometric_render_normal = select(
                -geometric_render_normal,
                geometric_render_normal,
                dot(geometric_render_normal, normal) >= 0.0,
            );
        }
        let dpdu = transform(transform_table.render_from_object, vec4<f32>(object_dpdu, 0.0)).xyz;
        let dpdv = transform(transform_table.render_from_object, vec4<f32>(object_dpdv, 0.0)).xyz;
        let dndu = transform(transform_table.normal_from_object, vec4<f32>(object_dndu, 0.0)).xyz;
        let dndv = transform(transform_table.normal_from_object, vec4<f32>(object_dndv, 0.0)).xyz;
        let differentials = uv_differentials(
            position,
            normal,
            dpdu,
            dpdv,
            rx_origin,
            rx_direction,
            ry_origin,
            ry_direction,
        );
        if ((material.flags & 2u) != 0u) {
            normal = apply_normal_map(
                material.normal_map,
                uv,
                normal,
                object_dpdu,
                object_tangent,
                transform_table.render_from_object,
            );
        } else if ((material.flags & 4u) != 0u) {
            normal = apply_bump_map(
                material.displacement,
                uv,
                normalize(transform(transform_table.normal_from_object, vec4<f32>(object_normal, 0.0)).xyz),
                dpdu,
                dpdv,
                dndu,
                dndv,
                differentials,
            );
        }
        var reflectance = material.reflectance.xyz;
        if ((material.flags & 1u) != 0u) {
            reflectance = sample_texture(material.texture, uv, differentials);
        }
        let hit_area_light_index = find_area_light(intersection.instance_custom_data, intersection.primitive_index);
        if (hit_area_light_index != 0xffffffffu) {
            let hit_area_light = lights[hit_area_light_index];
            let emits_toward_ray = (hit_area_light.flags & 1u) != 0u
                || dot(geometric_render_normal, -direction) >= 0.0;
            if (emits_toward_ray && area_light_alpha_accept(hit_area_light, uv, position)) {
                var emission_weight = 1.0;
                if (depth > 0u) {
                    let light_pdf = area_light_pdf(
                        hit_area_light,
                        previous_context_position,
                        previous_context_shading_normal,
                        position,
                        direction,
                    ) / f32(arrayLength(&lights));
                    emission_weight = previous_bsdf_pdf / (previous_bsdf_pdf + light_pdf);
                }
                color += throughput * hit_area_light.intensity.xyz * emission_weight;
            }
        }
        if (depth == camera.bvh_info.z) {
            break;
        }

        let ray_sample = independent_ray_sample(sample_pixel, sample_index, depth);
        let direct_sample = ray_sample.direct;
        let light_count = arrayLength(&lights);
        let light_index = min(u32(direct_sample.light_selection * f32(light_count)), light_count - 1u);
        let light = lights[light_index];
        if (light.kind == 0u) {
            let to_light = light.position.xyz - position;
            let distance_squared = max(dot(to_light, to_light), 1.0e-8);
            let wi = normalize(to_light);
            let cosine = abs(dot(normal, wi));
            let same_hemisphere = dot(normal, -direction) * dot(normal, wi) > 0.0;
            let shadow_origin = offset_ray_origin(
                position,
                position_error,
                geometric_render_normal,
                to_light,
            );
            if (same_hemisphere && cosine > 0.0 && shadow_visible(
                shadow_origin,
                to_light,
                intersection.instance_custom_data,
                intersection.primitive_index,
            )) {
                color += throughput * reflectance * light.intensity.xyz * cosine
                    * f32(light_count) / (3.141592653589793 * distance_squared);
            }
        } else if (light.kind == 2u) {
            let light_context_position = offset_ray_origin(
                position,
                position_error,
                geometric_render_normal,
                -direction,
            );
            let light_sample = sample_area_light(
                light,
                light_context_position,
                normal,
                direct_sample.light_sample,
            );
            if (light_sample.valid
                && area_light_alpha_accept(light, light_sample.uv, light_sample.position)) {
                let to_light = light_sample.position - light_context_position;
                let wi = normalize(to_light);
                let cosine = abs(dot(normal, wi));
                let same_hemisphere = dot(normal, -direction) * dot(normal, wi) > 0.0;
                let bsdf_pdf = select(
                    cosine / 3.141592653589793,
                    0.0,
                    (light.flags & 2u) != 0u,
                );
                let light_pdf = light_sample.pdf / f32(light_count);
                let shadow_origin = offset_ray_origin(
                    position,
                    position_error,
                    geometric_render_normal,
                    to_light,
                );
                if (same_hemisphere && cosine > 0.0 && shadow_visible(
                    shadow_origin,
                    light_sample.position - shadow_origin,
                    intersection.instance_custom_data,
                    intersection.primitive_index,
                )) {
                    color += throughput * reflectance * light.intensity.xyz * cosine
                        / (3.141592653589793 * (bsdf_pdf + light_pdf));
                }
            }

        }
        let disk = sample_uniform_disk_concentric(ray_sample.indirect.direction);
        var local_wi = vec3<f32>(disk, sqrt(max(0.0, 1.0 - dot(disk, disk))));
        if (dot(normal, -direction) < 0.0) {
            local_wi.z = -local_wi.z;
        }
        let projected_dpdu = dpdu - normal * dot(normal, dpdu);
        let frame_x = normalize(select(
            coordinate_tangent(normal),
            projected_dpdu,
            dot(projected_dpdu, projected_dpdu) > 1.0e-20,
        ));
        let frame_y = cross(normal, frame_x);
        let next_direction = normalize(
            frame_x * local_wi.x + frame_y * local_wi.y + normal * local_wi.z,
        );
        previous_context_position = offset_ray_origin(
            position,
            position_error,
            geometric_render_normal,
            -direction,
        );
        previous_context_shading_normal = normal;
        previous_bsdf_pdf = abs(dot(normal, next_direction)) / 3.141592653589793;

        throughput *= reflectance;
        if (depth >= 1u) {
            let maximum = max(throughput.x, max(throughput.y, throughput.z));
            if (maximum < 1.0) {
                let q = 1.0 - maximum;
                if (ray_sample.indirect.roulette < q) {
                    break;
                }
                throughput /= 1.0 - q;
            }
        }
        ray_origin = offset_ray_origin(
            position,
            position_error,
            geometric_render_normal,
            next_direction,
        );
        ray_direction = next_direction;
        ray_rx_origin = ray_origin;
        ray_ry_origin = ray_origin;
        ray_rx_direction = ray_direction;
        ray_ry_direction = ray_direction;
    } else {
        color += throughput * infinite_emission();
        break;
    }
    }

    return color;
}