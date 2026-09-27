@compute @workgroup_size(64, 1, 1)
fn sample_medium(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= atomicLoad(&queue_counters.medium_active.count)) { return; }
    let ray_index = active_medium_indices[queue_index];
    var ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let infinite = surface.hit == 0u;
    let distance = select(surface.t * length(ray.direction.xyz), 0.0, infinite);
    var weight = vec4<f32>(1.0);

    if (ray.medium_id != 0xffffffffu) {
        if (ray.medium_id >= arrayLength(&media)) {
            set_render_error();
            return;
        }
        let medium = media[ray.medium_id];
        if (medium.kind != 0u) {
            set_render_error();
            return;
        }
        let lambda = load_sample_lambda(pixel_index);
        let sigma_a = evaluate_spectrum(medium.sigma_a, lambda);
        if (sigma_a.x > 0.0) {
        if (infinite) {
            ray.beta = vec4<f32>(0.0);
            store_current_ray(ray_index, ray);
            return;
        }
        let u = random01(pixel_index, 16384u + ray.medium_segment_index, ray.depth);
        let event_distance = -log(1.0 - u) / sigma_a.x;
        if (event_distance < distance) {
            ray.beta = vec4<f32>(0.0);
            surfaces[pixel_index].hit = 0u;
            store_current_ray(ray_index, ray);
            return;
        }
        weight = exp((vec4<f32>(sigma_a.x) - sigma_a) * distance);
        } else if (sigma_a.x == 0.0) {
        if (infinite) {
            weight = select(vec4<f32>(0.0), vec4<f32>(1.0), sigma_a == vec4<f32>(0.0));
        } else {
            weight = exp(-distance * sigma_a);
        }
        } else {
            set_render_error();
            return;
        }
        ray.beta *= weight;
        ray.r_u *= weight;
        ray.r_l *= weight;
    }

    if (!any(ray.beta != vec4<f32>(0.0))) {
        surfaces[pixel_index].hit = 0u;
        store_current_ray(ray_index, ray);
        return;
    }

    if (surface.hit == 1u
        && instances[surface.instance_custom_data].material_root == 0xffffffffu) {
        let triangle = reconstruct_triangle_surface(
            surface.instance_custom_data,
            surface.primitive_index,
            vec3<f32>(1.0 - surface.barycentric.x - surface.barycentric.y,
                      surface.barycentric.x, surface.barycentric.y),
        );
        if (triangle.valid == 0u) { return; }
        var boundary = surface;
        boundary.position = vec4<f32>(triangle.position, 1.0);
        boundary.position_error = vec4<f32>(triangle.position_error, 0.0);
        boundary.geometric_normal = vec4<f32>(triangle.geometric_normal, 0.0);
        let segment_progress = dot(triangle.position - ray.origin.xyz, ray.direction.xyz);
        if (!(segment_progress > 0.0)) {
            set_render_error();
            return;
        }
        ray.medium_id = medium_for_direction(ray, boundary, ray.direction.xyz);
        ray.origin = vec4<f32>(offset_ray_origin(
            triangle.position, triangle.position_error,
            triangle.geometric_normal, ray.direction.xyz,
        ), 1.0);
        ray.medium_segment_index += 1u;
        surfaces[pixel_index].hit = 0u;
        append_medium_continuation(ray_index);
    }
    store_current_ray(ray_index, ray);
    if (infinite && any(ray.beta != vec4<f32>(0.0))) {
        append_escaped_ray(ray_index);
    }
}
