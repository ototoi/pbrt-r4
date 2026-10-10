fn sample_uniform_disk_concentric(u: vec2<f32>) -> vec2<f32> {
    let u_offset = 2.0 * u - vec2<f32>(1.0);
    if (u_offset.x == 0.0 && u_offset.y == 0.0) {
        return vec2<f32>(0.0);
    }
    if (abs(u_offset.x) > abs(u_offset.y)) {
        let radius = u_offset.x;
        let theta = 0.7853981633974483 * (u_offset.y / u_offset.x);
        return radius * vec2<f32>(cos(theta), sin(theta));
    }
    let radius = u_offset.y;
    let theta = 1.5707963267948966 - 0.7853981633974483 * (u_offset.x / u_offset.y);
    return radius * vec2<f32>(cos(theta), sin(theta));
}

@compute @workgroup_size(8, 8, 1)
fn generate_camera_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.tile_width || global_id.y >= viewport.tile_height) {
        return;
    }
    let pixel_index = global_id.y * viewport.tile_width + global_id.x;
    var wavelength_u = sampler_get_1d(pixel_index, 0u);
    if (viewport.disable_wavelength_jitter != 0u) { wavelength_u = 0.5; }
    let wavelength_offsets = fract(vec4<f32>(wavelength_u) + vec4<f32>(0.0, 0.25, 0.5, 0.75));
    let lambda = 538.0 - 138.888889 * atanh(vec4<f32>(0.85691062) - 1.82750197 * wavelength_offsets);
    let lambda_pdf = vec4<f32>(0.0039398042)
        / pow(cosh(0.0072 * (lambda - vec4<f32>(538.0))), vec4<f32>(2.0));
    store_sample_wavelengths(pixel_index, lambda, lambda_pdf);
    let jitter = select(
        sampler_get_pixel_2d(pixel_index), vec2<f32>(0.5),
        camera.disable_pixel_jitter != 0u,
    );
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, 0u));
    // camera.raster_to_camera expects a raster coordinate in the full image,
    // not a tile-local one.
    let pixel = vec4<f32>(
        f32(viewport.tile_x + global_id.x) + jitter.x - 0.5,
        f32(viewport.tile_y + global_id.y) + jitter.y - 0.5,
        1.0,
        1.0,
    );
    let camera_point = camera.raster_to_camera * pixel;
    var origin = (camera.camera_to_world * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    var direction = normalize((camera.camera_to_world * vec4<f32>(camera_point.xyz, 0.0)).xyz);
    if (camera.lens_radius > 0.0) {
        var lens_sample = vec2<f32>(0.5);
        if (camera.disable_pixel_jitter == 0u) {
            lens_sample = sampler_get_2d(pixel_index, 4u);
        }
        let lens_point = sample_uniform_disk_concentric(lens_sample) * camera.lens_radius;
        let camera_direction = normalize(camera_point.xyz);
        let focal_t = camera.focal_distance / camera_direction.z;
        let focus_point = focal_t * camera_direction;
        let camera_origin = vec3<f32>(lens_point, 0.0);
        let lens_direction = normalize(focus_point - camera_origin);
        origin = (camera.camera_to_world * vec4<f32>(camera_origin, 1.0)).xyz;
        direction = normalize((camera.camera_to_world * vec4<f32>(lens_direction, 0.0)).xyz);
    }
    let ray = RayWorkItem(
        vec4<f32>(origin, 1.0),
        vec4<f32>(direction, 0.0),
        vec4<f32>(1.0),
        vec4<f32>(1.0),
        vec4<f32>(1.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        pixel_index,
        0u,
        1.0,
        0.0,
        1u,
        camera.medium_id, 0u, 0u,
    );
    let current_queue_index = atomicAdd(&queue_counters.current.count, 1u);
    if (current_queue_index >= pixel_count()) {
        atomicStore(&queue_counters.current.overflow, 1u);
        return;
    }
    store_current_ray(current_queue_index, ray);
}
