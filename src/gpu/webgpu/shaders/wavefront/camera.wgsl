@compute @workgroup_size(1)
fn prepare_camera_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }
    arena.capacity = pixel_count();
    atomicStore(&arena.overflow, 0u);
}

@compute @workgroup_size(64)
fn generate_camera_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_index = global_id.x;
    let count = pixel_count();
    if (pixel_index >= count || pixel_index >= arena.capacity) {
        return;
    }

    let width = u32(camera.viewport.z);
    let local_pixel = vec2<u32>(pixel_index % width, pixel_index / width);
    let pixel = vec2<i32>(local_pixel) + vec2<i32>(camera.viewport.xy);
    let camera_sample = independent_camera_sample(
        pixel,
        camera.sampler_info.z + arena.sample_index,
    );
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

    arena.rays[pixel_index].origin = vec4<f32>(
        transform(camera.render_from_camera, vec4<f32>(camera_origin, 1.0)).xyz,
        0.0,
    );
    arena.rays[pixel_index].direction = vec4<f32>(normalize(transform(
        camera.render_from_camera,
        vec4<f32>(camera_direction, 0.0),
    ).xyz), 0.0);
    arena.rays[pixel_index].hit = vec4<f32>(0.0);
    arena.rays[pixel_index].indices = vec4<u32>(pixel_index, RAY_STATE_ACTIVE, 0u, 0u);
}
