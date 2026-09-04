@compute @workgroup_size(8, 8, 1)
fn generate_primary_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let pixel_index = global_id.y * viewport.width + global_id.x;
    let jitter = vec2<f32>(random01(pixel_index, 0u), random01(pixel_index, 1u));
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
        pixel_index,
        0u,
        1u,
        0u,
    );
    let camera_queue_index = atomicAdd(&camera_ray_queue_state.count, 1u);
    let current_queue_index = atomicAdd(&current_ray_queue_state.count, 1u);
    if (camera_queue_index >= camera_ray_queue_state.capacity
        || current_queue_index >= current_ray_queue_state.capacity) {
        atomicStore(&camera_ray_queue_state.overflow, 1u);
        atomicStore(&current_ray_queue_state.overflow, 1u);
        return;
    }
    camera_ray_queue[camera_queue_index] = ray;
    current_ray_queue[current_queue_index] = ray;
    // Keep the legacy queue populated until all stages have migrated to the
    // compact queues. This makes the new queue state observable without
    // changing the current renderer's image path yet.
    rays[pixel_index] = ray;
}
