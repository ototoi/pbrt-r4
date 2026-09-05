@compute @workgroup_size(8, 8, 1)
fn generate_primary_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let pixel_index = global_id.y * viewport.width + global_id.x;
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
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        pixel_index,
        0u,
        1.0,
        1.0,
        0.0,
        vec3<u32>(0u, 0u, 0u),
    );
    let current_queue_index = atomicAdd(&wavefront_queue[CURRENT_COUNT], 1u);
    if (current_queue_index >= pixel_count()) {
        atomicStore(&wavefront_queue[CURRENT_OVERFLOW], 1u);
        return;
    }
    store_current_ray(current_queue_index, ray);
}
