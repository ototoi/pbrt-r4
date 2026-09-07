@compute @workgroup_size(8, 8, 1)
fn generate_primary_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let pixel_index = global_id.y * viewport.width + global_id.x;
    var wavelength_u = random01(pixel_index, 8u, 0u);
    if (viewport.disable_wavelength_jitter != 0u) { wavelength_u = 0.5; }
    let wavelength_offsets = fract(vec4<f32>(wavelength_u) + vec4<f32>(0.0, 0.25, 0.5, 0.75));
    let lambda = 538.0 - 138.888889 * atanh(vec4<f32>(0.85691062) - 1.82750197 * wavelength_offsets);
    let lambda_pdf = vec4<f32>(0.0039398042)
        / pow(cosh(0.0072 * (lambda - vec4<f32>(538.0))), vec4<f32>(2.0));
    store_sample_wavelengths(pixel_index, lambda, lambda_pdf);
    let jitter = vec2<f32>(random01(pixel_index, 0u, 0u), random01(pixel_index, 1u, 0u));
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, 0u));
    let pixel = vec4<f32>(
        f32(global_id.x) + jitter.x - 0.5,
        f32(global_id.y) + jitter.y - 0.5,
        1.0,
        1.0,
    );
    let camera_point = camera.raster_to_camera * pixel;
    let origin = (camera.camera_to_world * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let direction = normalize((camera.camera_to_world * vec4<f32>(camera_point.xyz, 0.0)).xyz);
    let ray = RayWorkItem(
        vec4<f32>(origin, 1.0),
        vec4<f32>(direction, 0.0),
        vec4<f32>(1.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        pixel_index,
        0u,
        1.0,
        1.0,
        0.0,
        0u, 0u, 0u,
    );
    let current_queue_index = atomicAdd(&queue_counters.current.count, 1u);
    if (current_queue_index >= pixel_count()) {
        atomicStore(&queue_counters.current.overflow, 1u);
        return;
    }
    store_current_ray(current_queue_index, ray);
}
