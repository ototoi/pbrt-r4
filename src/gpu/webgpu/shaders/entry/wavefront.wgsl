fn wavefront_pixel_index(global_id: vec3<u32>) -> u32 {
    return global_id.x;
}

@compute @workgroup_size(1)
fn prepare_camera_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u) {
        let pixel_count = u32(camera.viewport.z * camera.viewport.w);
        wavefront.ray_queue_a.capacity = pixel_count;
        wavefront.ray_queue_b.capacity = pixel_count;
        wavefront.escaped_ray_queue.capacity = pixel_count;
        wavefront.hit_area_light_queue.capacity = pixel_count;
        wavefront.surface_queue.capacity = pixel_count;
        wavefront.ray_queue_a.offset = 0u;
        wavefront.ray_queue_b.offset = pixel_count;
        wavefront.escaped_ray_queue.offset = pixel_count * 2u;
        wavefront.hit_area_light_queue.offset = pixel_count * 3u;
        wavefront.surface_queue.offset = pixel_count * 4u;
        wavefront.shadow_queue.capacity = pixel_count;
        wavefront.shadow_queue.offset = pixel_count * 5u;
        atomicStore(&wavefront.ray_queue_a.count, pixel_count);
        atomicStore(&wavefront.ray_queue_b.count, 0u);
        atomicStore(&wavefront.escaped_ray_queue.count, 0u);
        atomicStore(&wavefront.hit_area_light_queue.count, 0u);
        atomicStore(&wavefront.surface_queue.count, 0u);
        atomicStore(&wavefront.shadow_queue.count, 0u);
        atomicStore(&wavefront.ray_queue_a.overflow, 0u);
        atomicStore(&wavefront.ray_queue_b.overflow, 0u);
        atomicStore(&wavefront.escaped_ray_queue.overflow, 0u);
        atomicStore(&wavefront.hit_area_light_queue.overflow, 0u);
        atomicStore(&wavefront.surface_queue.overflow, 0u);
        atomicStore(&wavefront.shadow_queue.overflow, 0u);
        atomicStore(&wavefront.control.active_count, pixel_count);
        atomicStore(&wavefront.control.next_count, 0u);
        atomicStore(&wavefront.control.overflow, 0u);
    }
}

@compute @workgroup_size(64)
fn generate_camera_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = wavefront_pixel_index(global_id);
    let width = u32(camera.viewport.z);
    let height = u32(camera.viewport.w);
    let pixel_count = width * height;
    if (index >= pixel_count) {
        return;
    }

    let local_pixel = vec2<u32>(index % width, index / width);
    let pixel = vec2<i32>(local_pixel) + vec2<i32>(camera.viewport.xy);
    let sample_index = camera.sampler_info.z + wavefront.control.sample_index;
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
    wavefront_store_queue_index(index, index);
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
    wavefront.slots[index].surface_position = vec4<f32>(0.0);
    wavefront.slots[index].surface_position_error = vec4<f32>(0.0);
    wavefront.slots[index].surface_geometric_normal = vec4<f32>(0.0);
    wavefront.slots[index].surface_shading_normal = vec4<f32>(0.0);
    wavefront.slots[index].surface_uv = vec4<f32>(0.0);
    wavefront.slots[index].previous_surface_position = vec4<f32>(0.0);
    wavefront.slots[index].previous_surface_shading_normal = vec4<f32>(0.0);
    wavefront.slots[index].previous_bsdf_pdf = vec4<f32>(0.0);
    wavefront.slots[index].material_parameters = vec4<f32>(0.0);
}

@compute @workgroup_size(64)
fn intersect_closest(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.ray_queue_a.count)
        || queue_index >= wavefront.ray_queue_a.capacity) {
        return;
    }
    let index = wavefront_queue_index(wavefront.ray_queue_a.offset + queue_index);
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

struct SurfaceData {
    position: vec3<f32>,
    position_error: vec3<f32>,
    geometric_normal: vec3<f32>,
    shading_normal: vec3<f32>,
    uv: vec2<f32>,
    primitive_id: u32,
    material_id: u32,
    pixel: vec2<i32>,
    depth: u32,
};

fn load_surface_data(index: u32) -> SurfaceData {
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
    let uv = vertices[i0].uv.xy * barycentrics.x
        + vertices[i1].uv.xy * barycentrics.y
        + vertices[i2].uv.xy * barycentrics.z;
    var shading_normal = geometric_normal;
    if (dot(object_normal, object_normal) > 1.0e-20) {
        shading_normal = normalize(transform(
            transform_table.normal_from_object,
            vec4<f32>(object_normal, 0.0),
        ).xyz);
        shading_normal = select(
            -shading_normal,
            shading_normal,
            dot(shading_normal, geometric_normal) >= 0.0,
        );
    }
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
    let width = u32(camera.viewport.z);
    let local_pixel = vec2<u32>(index % width, index / width);
    return SurfaceData(
        position,
        position_error,
        geometric_normal,
        shading_normal,
        uv,
        primitive_index,
        primitive.material,
        vec2<i32>(local_pixel) + vec2<i32>(camera.viewport.xy),
        wavefront.slots[index].path_info.x,
    );
}

@compute @workgroup_size(64)
fn classify_intersection(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.ray_queue_a.count)
        || queue_index >= wavefront.ray_queue_a.capacity) {
        return;
    }
    let index = wavefront_queue_index(wavefront.ray_queue_a.offset + queue_index);
    if (index >= arrayLength(&wavefront.slots)) {
        return;
    }
    let state = wavefront.slots[index].intersection_ids.x;
    if (state == WAVEFRONT_STATE_HIT) {
        for (var light_index = 0u; light_index < light_source_count(); light_index++) {
            let light = lights[light_index];
            if (light.kind == 2u) {
                for (var geometry_index = 0u; geometry_index < light.triangle; geometry_index++) {
                    let geometry = area_light_geometry(light, geometry_index);
                    if (geometry.primitive == wavefront.slots[index].intersection_ids.y
                        && geometry.triangle == wavefront.slots[index].intersection_ids.z) {
                        wavefront.slots[index].intersection_ids.x = WAVEFRONT_STATE_EMISSIVE;
                        wavefront.slots[index].intersection_ids.w = light_index;
                        wavefront_enqueue_hit_area_light(index);
                        // Emissive hits also need the evaluated surface normal
                        // for sidedness and MIS, but are rejected by the BxDF
                        // stages because their state is EMISSIVE.
                        wavefront_enqueue_surface(index);
                        return;
                    }
                }
            }
        }
        wavefront_enqueue_surface(index);
        return;
    }
    if (state == WAVEFRONT_STATE_MISS) {
        wavefront_enqueue_escaped_ray(index);
        return;
    }
    if (state == WAVEFRONT_STATE_EMISSIVE) {
        wavefront_enqueue_hit_area_light(index);
        return;
    }
    if (state != WAVEFRONT_STATE_HIT && state != WAVEFRONT_STATE_MISS) {
        wavefront.slots[index].intersection_ids.x = WAVEFRONT_STATE_INACTIVE;
    }
}

@compute @workgroup_size(64)
fn evaluate_surface_interaction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.surface_queue.count)
        || queue_index >= wavefront.surface_queue.capacity) {
        return;
    }
    let index = wavefront_surface_queue_index(queue_index);
    if (index < arrayLength(&wavefront.slots)
        && (wavefront.slots[index].intersection_ids.x == WAVEFRONT_STATE_HIT
            || wavefront.slots[index].intersection_ids.x == WAVEFRONT_STATE_EMISSIVE)) {
        let surface = load_surface_data(index);
        wavefront.slots[index].surface_position = vec4<f32>(surface.position, 1.0);
        wavefront.slots[index].surface_position_error =
            vec4<f32>(surface.position_error, 0.0);
        wavefront.slots[index].surface_geometric_normal =
            vec4<f32>(surface.geometric_normal, 0.0);
        wavefront.slots[index].surface_shading_normal =
            vec4<f32>(surface.shading_normal, 0.0);
        wavefront.slots[index].surface_uv = vec4<f32>(surface.uv, 0.0, 0.0);
    }
}

@compute @workgroup_size(64)
fn register_bxdf(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.surface_queue.count)
        || queue_index >= wavefront.surface_queue.capacity) {
        return;
    }
    let index = wavefront_surface_queue_index(queue_index);
    if (index < arrayLength(&wavefront.slots)
        && wavefront.slots[index].intersection_ids.x == WAVEFRONT_STATE_HIT) {
        // BxDF kind 0 is diffuse in the initial GPU material set.
        wavefront.slots[index].path_info.z = 0u;
        wavefront.slots[index].path_info.w = 1u;
    }
}

@compute @workgroup_size(64)
fn count_bxdf(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.surface_queue.count)
        || queue_index >= wavefront.surface_queue.capacity) {
        return;
    }
    let index = wavefront_surface_queue_index(queue_index);
    if (index < arrayLength(&wavefront.slots)
        && wavefront.slots[index].intersection_ids.x == WAVEFRONT_STATE_HIT) {
        // One diffuse BxDF work item is emitted for this surface.
        wavefront.slots[index].path_info.w = 1u;
    }
}

@compute @workgroup_size(64)
fn partition_bxdf(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.surface_queue.count)
        || queue_index >= wavefront.surface_queue.capacity) {
        return;
    }
    let index = wavefront_surface_queue_index(queue_index);
    if (index < arrayLength(&wavefront.slots)
        && wavefront.slots[index].intersection_ids.x == WAVEFRONT_STATE_HIT) {
        // Diffuse is the only supported BxDF in the first WebGPU slice, so the
        // one-item range is already partitioned in place.
    }
}

@compute @workgroup_size(64)
fn sample_direct_lighting(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.surface_queue.count)
        || queue_index >= wavefront.surface_queue.capacity) {
        return;
    }
    let index = wavefront_surface_queue_index(queue_index);
    if (index < arrayLength(&wavefront.slots)
        && wavefront.slots[index].intersection_ids.x == WAVEFRONT_STATE_HIT
        && wavefront.slots[index].path_info.z == 0u
        && wavefront.slots[index].path_info.w > 0u
        && wavefront.slots[index].path_info.x < camera.bvh_info.z) {
        let position = wavefront.slots[index].surface_position.xyz;
        let position_error = wavefront.slots[index].surface_position_error.xyz;
        let geometric_normal = wavefront.slots[index].surface_geometric_normal.xyz;
        let normal = wavefront.slots[index].surface_shading_normal.xyz;
        let wo = -wavefront.slots[index].ray_direction.xyz;
        let depth = wavefront.slots[index].path_info.x;
        let ray_sample = independent_ray_sample(
            vec2<i32>(
                i32(index % u32(camera.viewport.z)),
                i32(index / u32(camera.viewport.z)),
            ) + vec2<i32>(camera.viewport.xy),
            camera.sampler_info.z + wavefront.control.sample_index,
            depth,
        );
        let reflectance = wavefront.slots[index].material_parameters.xyz;
        wavefront.slots[index].shadow_origin = vec4<f32>(0.0);
        wavefront.slots[index].contribution = vec4<f32>(0.0);
        let light_count = light_source_count();
        if (light_count > 0u) {
            let light_index = min(
                u32(ray_sample.direct.light_selection * f32(light_count)),
                light_count - 1u,
            );
            let light = lights[light_index];
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
                            * light.intensity.xyz * cosine * f32(light_count)
                            / (3.141592653589793 * distance_squared),
                        1.0,
                    );
                    wavefront_enqueue_shadow(index);
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
                        (light.flags & 2u) != 0u,
                    );
                    let light_pdf = light_sample.pdf / f32(light_count);
                    if (same_hemisphere && cosine > 0.0 && light_pdf > 0.0) {
                        wavefront.slots[index].shadow_origin = vec4<f32>(context_position, 1.0);
                        wavefront.slots[index].shadow_direction = vec4<f32>(to_light, 0.0);
                        wavefront.slots[index].contribution = vec4<f32>(
                            wavefront.slots[index].throughput.xyz * reflectance
                                * light.intensity.xyz * cosine
                                / (3.141592653589793 * (bsdf_pdf + light_pdf)),
                            1.0,
                        );
                        wavefront_enqueue_shadow(index);
                    }
                }
            } else if (light.kind == 1u) {
                let z = 1.0 - 2.0 * ray_sample.direct.light_sample.x;
                let phi = 6.283185307179586 * ray_sample.direct.light_sample.y;
                let radial = sqrt(max(0.0, 1.0 - z * z));
                let wi = vec3<f32>(radial * cos(phi), radial * sin(phi), z);
                let cosine = abs(dot(normal, wi));
                let same_hemisphere = dot(normal, wo) * dot(normal, wi) > 0.0;
                let light_pdf = 0.07957747154594767 / f32(light_count);
                let bsdf_pdf = cosine / 3.141592653589793;
                if (same_hemisphere && cosine > 0.0 && light_pdf > 0.0) {
                    let shadow_origin = offset_ray_origin(
                        position,
                        position_error,
                        geometric_normal,
                        wi,
                    );
                    wavefront.slots[index].shadow_origin = vec4<f32>(shadow_origin, 1.0);
                    wavefront.slots[index].shadow_direction = vec4<f32>(wi, 1.0);
                    wavefront.slots[index].contribution = vec4<f32>(
                        wavefront.slots[index].throughput.xyz
                            * reflectance
                            * light.intensity.xyz
                            * cosine
                            / (3.141592653589793 * (bsdf_pdf + light_pdf)),
                        1.0,
                    );
                    wavefront_enqueue_shadow(index);
                }
            }
        }
    }
}

@compute @workgroup_size(64)
fn generate_indirect_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.surface_queue.count)
        || queue_index >= wavefront.surface_queue.capacity) {
        return;
    }
    let index = wavefront_surface_queue_index(queue_index);
    if (index < arrayLength(&wavefront.slots)
        && wavefront.slots[index].intersection_ids.x == WAVEFRONT_STATE_HIT
        && wavefront.slots[index].path_info.z == 0u
        && wavefront.slots[index].path_info.w > 0u) {
        if (wavefront.slots[index].path_info.x >= camera.bvh_info.z) {
            wavefront.slots[index].path_info.y = 0u;
            return;
        }
        let position = wavefront.slots[index].surface_position.xyz;
        let position_error = wavefront.slots[index].surface_position_error.xyz;
        let geometric_normal = wavefront.slots[index].surface_geometric_normal.xyz;
        let normal = wavefront.slots[index].surface_shading_normal.xyz;
        let wo = -wavefront.slots[index].ray_direction.xyz;
        let depth = wavefront.slots[index].path_info.x;
        let ray_sample = independent_ray_sample(
            vec2<i32>(
                i32(index % u32(camera.viewport.z)),
                i32(index / u32(camera.viewport.z)),
            ) + vec2<i32>(camera.viewport.xy),
            camera.sampler_info.z + wavefront.control.sample_index,
            depth,
        );
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
        wavefront.slots[index].previous_surface_position = vec4<f32>(position, 1.0);
        wavefront.slots[index].previous_surface_shading_normal = vec4<f32>(normal, 0.0);
        wavefront.slots[index].previous_bsdf_pdf = vec4<f32>(
            abs(dot(normal, next_direction)) / 3.141592653589793,
            0.0,
            0.0,
            0.0,
        );
        let reflectance = wavefront.slots[index].material_parameters.xyz;
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
}

@compute @workgroup_size(64)
fn handle_escaped_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.escaped_ray_queue.count)
        || queue_index >= wavefront.escaped_ray_queue.capacity) {
        return;
    }
    let index = wavefront_escaped_ray_queue_index(queue_index);
    if (index < arrayLength(&wavefront.slots)
        && wavefront.slots[index].intersection_ids.x == WAVEFRONT_STATE_MISS) {
        wavefront.slots[index].contribution = vec4<f32>(
            wavefront.slots[index].throughput.xyz * infinite_emission(),
            1.0,
        );
        wavefront.slots[index].path_info.y = 0u;
    }
}

@compute @workgroup_size(64)
fn handle_emissive_intersection(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.hit_area_light_queue.count)
        || queue_index >= wavefront.hit_area_light_queue.capacity) {
        return;
    }
    let index = wavefront_hit_area_light_queue_index(queue_index);
    if (index < arrayLength(&wavefront.slots)
        && wavefront.slots[index].intersection_ids.x == WAVEFRONT_STATE_EMISSIVE) {
        let light = lights[wavefront.slots[index].intersection_ids.w];
        let emits_toward_ray = (light.flags & 1u) != 0u
            || dot(
                wavefront.slots[index].surface_geometric_normal.xyz,
                -wavefront.slots[index].ray_direction.xyz,
            ) >= 0.0;
        if (emits_toward_ray) {
            var emission_weight = 1.0;
            if (wavefront.slots[index].path_info.x > 0u
                && wavefront.slots[index].previous_bsdf_pdf.x > 0.0) {
                let light_count = light_source_count();
                let light_pdf = area_light_pdf(
                    light,
                    wavefront.slots[index].previous_surface_position.xyz,
                    wavefront.slots[index].previous_surface_shading_normal.xyz,
                    wavefront.slots[index].surface_position.xyz,
                    wavefront.slots[index].ray_direction.xyz,
                    wavefront.slots[index].intersection_ids.z,
                ) / f32(light_count);
                emission_weight = wavefront.slots[index].previous_bsdf_pdf.x
                    / (wavefront.slots[index].previous_bsdf_pdf.x + light_pdf);
            }
            wavefront.slots[index].contribution = vec4<f32>(
                wavefront.slots[index].throughput.xyz * light.intensity.xyz * emission_weight,
                1.0,
            );
        } else {
            wavefront.slots[index].contribution = vec4<f32>(0.0);
        }
        wavefront.slots[index].path_info.y = 0u;
    }
}

@compute @workgroup_size(64)
fn evaluate_material(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.surface_queue.count)
        || queue_index >= wavefront.surface_queue.capacity) {
        return;
    }
    let index = wavefront_surface_queue_index(queue_index);
    if (index >= arrayLength(&wavefront.slots)) {
        return;
    }
    if (wavefront.slots[index].intersection_ids.x != WAVEFRONT_STATE_HIT) {
        wavefront.slots[index].shadow_origin = vec4<f32>(0.0);
        return;
    }
    let primitive_index = wavefront.slots[index].intersection_ids.y;
    let primitive = primitives[primitive_index];
    let material = materials[primitive.material];
    var reflectance = material.reflectance.xyz;
    if ((material.flags & 1u) != 0u) {
        reflectance = sample_texture(
            material.texture,
            wavefront.slots[index].surface_uv.xy,
            vec4<f32>(0.0),
        );
    }
    wavefront.slots[index].material_parameters = vec4<f32>(reflectance, 1.0);
    wavefront.slots[index].shadow_origin = vec4<f32>(0.0);
    wavefront.slots[index].contribution = vec4<f32>(0.0);
}

@compute @workgroup_size(64)
fn intersect_shadow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.shadow_queue.count)
        || queue_index >= wavefront.shadow_queue.capacity) {
        return;
    }
    let index = wavefront_shadow_queue_index(queue_index);
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
            select(0.9999, 1.0e30, wavefront.slots[index].shadow_direction.w != 0.0),
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
    let queue_index = wavefront_pixel_index(global_id);
    if (queue_index >= atomicLoad(&wavefront.ray_queue_a.count)
        || queue_index >= wavefront.ray_queue_a.capacity) {
        return;
    }
    let index = wavefront_queue_index(wavefront.ray_queue_a.offset + queue_index);
    if (index >= arrayLength(&wavefront.slots)) {
        return;
    }
    wavefront.slots[index].radiance += wavefront.slots[index].contribution;
    wavefront.slots[index].contribution = vec4<f32>(0.0);
    wavefront.slots[index].shadow_origin = vec4<f32>(0.0);
    if (wavefront.slots[index].path_info.y != 0u) {
        wavefront.slots[index].path_info.x += 1u;
        wavefront.slots[index].intersection_ids.x = WAVEFRONT_STATE_RAY;
        let next_index = atomicAdd(&wavefront.ray_queue_b.count, 1u);
        if (next_index >= wavefront.ray_queue_b.capacity) {
            atomicStore(&wavefront.ray_queue_b.overflow, 1u);
            atomicStore(&wavefront.control.overflow, 1u);
        } else {
            wavefront_store_queue_index(wavefront.ray_queue_b.offset + next_index, index);
        }
    } else {
        wavefront.slots[index].intersection_ids.x = WAVEFRONT_STATE_INACTIVE;
    }
}

@compute @workgroup_size(1)
fn prepare_next_bounce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u) {
        let next_count = min(
            atomicLoad(&wavefront.ray_queue_b.count),
            wavefront.ray_queue_b.capacity,
        );
        atomicStore(&wavefront.ray_queue_a.count, next_count);
        atomicStore(&wavefront.ray_queue_a.overflow, 0u);
        let previous_offset = wavefront.ray_queue_a.offset;
        wavefront.ray_queue_a.offset = wavefront.ray_queue_b.offset;
        wavefront.ray_queue_b.offset = previous_offset;
        atomicStore(&wavefront.ray_queue_b.count, 0u);
        atomicStore(&wavefront.ray_queue_b.overflow, 0u);
        atomicStore(&wavefront.escaped_ray_queue.count, 0u);
        atomicStore(&wavefront.escaped_ray_queue.overflow, 0u);
        atomicStore(&wavefront.hit_area_light_queue.count, 0u);
        atomicStore(&wavefront.hit_area_light_queue.overflow, 0u);
        atomicStore(&wavefront.surface_queue.count, 0u);
        atomicStore(&wavefront.surface_queue.overflow, 0u);
        atomicStore(&wavefront.shadow_queue.count, 0u);
        atomicStore(&wavefront.shadow_queue.overflow, 0u);
        atomicStore(&wavefront.control.active_count, next_count);
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
        wavefront.control.sample_index += 1u;
    }
}
