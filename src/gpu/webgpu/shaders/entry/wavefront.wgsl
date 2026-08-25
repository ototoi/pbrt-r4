fn wavefront_pixel_index(global_id: vec3<u32>) -> u32 {
    return global_id.x;
}

@compute @workgroup_size(64)
fn generate_camera(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = wavefront_pixel_index(global_id);
    let width = u32(camera.viewport.z);
    let height = u32(camera.viewport.w);
    let pixel_count = width * height;
    if (index >= pixel_count) {
        return;
    }

    let local_pixel = vec2<u32>(index % width, index / width);
    let pixel = vec2<i32>(local_pixel) + vec2<i32>(camera.viewport.xy);
    let sample_index = camera.sampler_info.z + wavefront.control.x;
    let camera_sample = independent_camera_sample(pixel, sample_index);
    let filter_offset = mix(-camera.filter_info.xy, camera.filter_info.xy, camera_sample.filter_sample);
    let film_position = vec2<f32>(pixel) + filter_offset + vec2<f32>(0.5);
    let camera_target = transform(
        camera.camera_from_raster,
        vec4<f32>(film_position, 0.0, 1.0),
    );
    var camera_origin = vec3<f32>(0.0);
    var camera_direction = normalize(camera_target.xyz);
    if (camera.camera_info.x > 0.0) {
        let lens = camera.camera_info.x * sample_uniform_disk_concentric(camera_sample.lens);
        let focus_t = camera.camera_info.y / camera_direction.z;
        let focus = focus_t * camera_direction;
        camera_origin = vec3<f32>(lens, 0.0);
        camera_direction = normalize(focus - camera_origin);
    }

    let ray_origin = transform(camera.render_from_camera, vec4<f32>(camera_origin, 1.0)).xyz;
    let ray_direction = normalize(transform(
        camera.render_from_camera,
        vec4<f32>(camera_direction, 0.0),
    ).xyz);
    wavefront.slots[index].ray_origin = vec4<f32>(ray_origin, 0.0);
    wavefront.slots[index].ray_direction = vec4<f32>(ray_direction, 0.0);
    wavefront.slots[index].intersection_ids = vec4<u32>(WAVEFRONT_STATE_RAY, 0u, 0u, 0u);
    wavefront.slots[index].intersection_data = vec4<f32>(0.0);
    wavefront.slots[index].shadow_origin = vec4<f32>(0.0);
    wavefront.slots[index].shadow_direction = vec4<f32>(0.0);
    wavefront.slots[index].contribution = vec4<f32>(0.0);
    wavefront.slots[index].radiance = vec4<f32>(0.0);
    wavefront.slots[index].throughput = vec4<f32>(1.0);
    wavefront.slots[index].path_info = vec4<u32>(0u);
}

@compute @workgroup_size(64)
fn intersect_closest(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = wavefront_pixel_index(global_id);
    if (index >= arrayLength(&wavefront.slots)
        || wavefront.slots[index].intersection_ids.x != WAVEFRONT_STATE_RAY) {
        return;
    }

    let origin = wavefront.slots[index].ray_origin.xyz;
    let direction = wavefront.slots[index].ray_direction.xyz;
    var query: ray_query;
    rayQueryInitialize(
        &query,
        acceleration,
        RayDesc(0u, 0xFFu, 0.0001, 1.0e30, origin, direction),
    );
    while (rayQueryProceed(&query)) {
        rayQueryConfirmIntersection(&query);
    }
    let intersection = rayQueryGetCommittedIntersection(&query);
    if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
        wavefront.slots[index].intersection_ids = vec4<u32>(WAVEFRONT_STATE_MISS, 0u, 0u, 0u);
        wavefront.slots[index].intersection_data = vec4<f32>(0.0);
        return;
    }

    wavefront.slots[index].intersection_ids = vec4<u32>(
        WAVEFRONT_STATE_HIT,
        intersection.instance_custom_data,
        intersection.primitive_index,
        intersection.geometry_index,
    );
    wavefront.slots[index].intersection_data = vec4<f32>(
        intersection.t,
        intersection.barycentrics,
        0.0,
    );
}

@compute @workgroup_size(64)
fn shade_diffuse_point(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = wavefront_pixel_index(global_id);
    if (index >= arrayLength(&wavefront.slots)) {
        return;
    }
    if (wavefront.slots[index].intersection_ids.x != WAVEFRONT_STATE_HIT) {
        if (wavefront.slots[index].intersection_ids.x == WAVEFRONT_STATE_MISS) {
            wavefront.slots[index].contribution = vec4<f32>(
                wavefront.slots[index].throughput.xyz * infinite_emission(),
                1.0,
            );
        } else {
            wavefront.slots[index].contribution = vec4<f32>(0.0);
        }
        wavefront.slots[index].shadow_origin = vec4<f32>(0.0);
        wavefront.slots[index].path_info.y = 0u;
        return;
    }

    let primitive_index = wavefront.slots[index].intersection_ids.y;
    let triangle_index = wavefront.slots[index].intersection_ids.z;
    let primitive = primitives[primitive_index];
    let index_offset = primitive.first_index + triangle_index * 3u;
    let i0 = primitive.first_vertex + indices[index_offset];
    let i1 = primitive.first_vertex + indices[index_offset + 1u];
    let i2 = primitive.first_vertex + indices[index_offset + 2u];
    let barycentrics = vec3<f32>(
        1.0 - wavefront.slots[index].intersection_data.y
            - wavefront.slots[index].intersection_data.z,
        wavefront.slots[index].intersection_data.y,
        wavefront.slots[index].intersection_data.z,
    );
    let transform_table = transforms[primitive_index];
    let object_position = vertices[i0].position.xyz * barycentrics.x
        + vertices[i1].position.xyz * barycentrics.y
        + vertices[i2].position.xyz * barycentrics.z;
    let position = transform(
        transform_table.render_from_object,
        vec4<f32>(object_position, 1.0),
    ).xyz;
    let render_p0 = transform(transform_table.render_from_object, vertices[i0].position).xyz;
    let render_p1 = transform(transform_table.render_from_object, vertices[i1].position).xyz;
    let render_p2 = transform(transform_table.render_from_object, vertices[i2].position).xyz;
    var geometric_normal = normalize(cross(render_p1 - render_p0, render_p2 - render_p0));
    if ((primitive.flags & 1u) != 0u) {
        geometric_normal = -geometric_normal;
    }
    let object_normal = vertices[i0].normal.xyz * barycentrics.x
        + vertices[i1].normal.xyz * barycentrics.y
        + vertices[i2].normal.xyz * barycentrics.z;
    var normal = geometric_normal;
    if (dot(object_normal, object_normal) > 1.0e-20) {
        normal = normalize(transform(
            transform_table.normal_from_object,
            vec4<f32>(object_normal, 0.0),
        ).xyz);
        normal = select(-normal, normal, dot(normal, geometric_normal) >= 0.0);
    }

    let wo = -wavefront.slots[index].ray_direction.xyz;
    let width = u32(camera.viewport.z);
    let local_pixel = vec2<u32>(index % width, index / width);
    let pixel = vec2<i32>(local_pixel) + vec2<i32>(camera.viewport.xy);
    let sample_index = camera.sampler_info.z + wavefront.control.x;
    let depth = wavefront.slots[index].path_info.x;
    let ray_sample = independent_ray_sample(pixel, sample_index, depth);
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
    let reflectance = materials[primitive.material].reflectance.xyz;
    let light = lights[0];
    wavefront.slots[index].shadow_origin = vec4<f32>(0.0);
    wavefront.slots[index].contribution = vec4<f32>(0.0);
    if (light.kind == 0u) {
        let to_light = light.position.xyz - position;
        let distance_squared = max(dot(to_light, to_light), 1.0e-8);
        let wi = normalize(to_light);
        let cosine = abs(dot(normal, wi));
        let same_hemisphere = dot(normal, wo) * dot(normal, wi) > 0.0;
        if (same_hemisphere && cosine > 0.0) {
            let shadow_origin = offset_ray_origin(
                position,
                position_error,
                geometric_normal,
                to_light,
            );
            wavefront.slots[index].shadow_origin = vec4<f32>(shadow_origin, 1.0);
            wavefront.slots[index].shadow_direction = vec4<f32>(to_light, 0.0);
            wavefront.slots[index].contribution = vec4<f32>(
                wavefront.slots[index].throughput.xyz * reflectance
                    * light.intensity.xyz * cosine
                    / (3.141592653589793 * distance_squared),
                1.0,
            );
        }
    } else if (light.kind == 2u) {
        let context_position = offset_ray_origin(
            position,
            position_error,
            geometric_normal,
            -wo,
        );
        let light_sample = sample_area_light(
            light,
            context_position,
            normal,
            ray_sample.direct.light_sample,
        );
        if (light_sample.valid) {
            let to_light = light_sample.position - context_position;
            let wi = normalize(to_light);
            let cosine = abs(dot(normal, wi));
            let same_hemisphere = dot(normal, wo) * dot(normal, wi) > 0.0;
            let bsdf_pdf = select(
                cosine / 3.141592653589793,
                0.0,
                (light.flags & 1u) != 0u,
            );
            let light_pdf = light_sample.pdf;
            if (same_hemisphere && cosine > 0.0 && light_pdf > 0.0) {
                wavefront.slots[index].shadow_origin = vec4<f32>(context_position, 1.0);
                wavefront.slots[index].shadow_direction = vec4<f32>(to_light, 0.0);
                wavefront.slots[index].contribution = vec4<f32>(
                    wavefront.slots[index].throughput.xyz * reflectance
                        * light.intensity.xyz * cosine
                        / (3.141592653589793 * (bsdf_pdf + light_pdf)),
                    1.0,
                );
            }
        }
    }
    let disk = sample_uniform_disk_concentric(ray_sample.indirect.direction);
    var local_wi = vec3<f32>(disk, sqrt(max(0.0, 1.0 - dot(disk, disk))));
    if (dot(normal, wo) < 0.0) {
        local_wi.z = -local_wi.z;
    }
    let frame_x = coordinate_tangent(normal);
    let frame_y = cross(normal, frame_x);
    let next_direction = normalize(
        frame_x * local_wi.x + frame_y * local_wi.y + normal * local_wi.z,
    );
    var next_throughput = wavefront.slots[index].throughput.xyz * reflectance;
    var continue_path = true;
    if (depth >= 1u) {
        let maximum = max(next_throughput.x, max(next_throughput.y, next_throughput.z));
        if (maximum < 1.0) {
            let q = 1.0 - maximum;
            if (ray_sample.indirect.roulette < q) {
                continue_path = false;
            } else {
                next_throughput /= 1.0 - q;
            }
        }
    }
    wavefront.slots[index].ray_origin = vec4<f32>(offset_ray_origin(
        position,
        position_error,
        geometric_normal,
        next_direction,
    ), 0.0);
    wavefront.slots[index].ray_direction = vec4<f32>(next_direction, 0.0);
    wavefront.slots[index].throughput = vec4<f32>(next_throughput, 1.0);
    wavefront.slots[index].path_info.y = select(0u, 1u, continue_path);
}

@compute @workgroup_size(64)
fn intersect_shadow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = wavefront_pixel_index(global_id);
    if (index >= arrayLength(&wavefront.slots)
        || wavefront.slots[index].shadow_origin.w == 0.0) {
        return;
    }

    let source_primitive = wavefront.slots[index].intersection_ids.y;
    let source_triangle = wavefront.slots[index].intersection_ids.z;
    var query: ray_query;
    rayQueryInitialize(
        &query,
        acceleration,
        RayDesc(
            0u,
            0xFFu,
            0.0,
            0.9999,
            wavefront.slots[index].shadow_origin.xyz,
            wavefront.slots[index].shadow_direction.xyz,
        ),
    );
    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        if (candidate.instance_custom_data == source_primitive
            && candidate.primitive_index == source_triangle) {
            continue;
        }
        rayQueryConfirmIntersection(&query);
    }
    if (rayQueryGetCommittedIntersection(&query).kind != RAY_QUERY_INTERSECTION_NONE) {
        wavefront.slots[index].contribution = vec4<f32>(0.0);
    }
}

@compute @workgroup_size(64)
fn finish_bounce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = wavefront_pixel_index(global_id);
    if (index >= arrayLength(&wavefront.slots)) {
        return;
    }
    wavefront.slots[index].radiance += wavefront.slots[index].contribution;
    wavefront.slots[index].contribution = vec4<f32>(0.0);
    wavefront.slots[index].shadow_origin = vec4<f32>(0.0);
    if (wavefront.slots[index].path_info.y != 0u) {
        wavefront.slots[index].path_info.x += 1u;
        wavefront.slots[index].intersection_ids.x = WAVEFRONT_STATE_RAY;
    } else {
        wavefront.slots[index].intersection_ids.x = WAVEFRONT_STATE_INACTIVE;
    }
}

@compute @workgroup_size(64)
fn update_film(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = wavefront_pixel_index(global_id);
    if (index >= arrayLength(&wavefront.slots)) {
        return;
    }
    output[index] += vec4<f32>(
        wavefront.slots[index].radiance.xyz / f32(camera.sampler_info.w),
        1.0 / f32(camera.sampler_info.w),
    );
}

@compute @workgroup_size(1)
fn advance_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u) {
        wavefront.control.x += 1u;
    }
}
