@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = u32(camera.viewport.z);
    let height = u32(camera.viewport.w);
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }
    let pixel = vec2<i32>(global_id.xy) + vec2<i32>(camera.viewport.xy);
    var accumulated = vec3<f32>(0.0);
    for (var local_sample = 0u; local_sample < camera.sampler_info.w; local_sample++) {
        let sample_index = camera.sampler_info.z + local_sample;
        let camera_sample = independent_camera_sample(pixel, sample_index);
        let filter_offset = mix(-camera.filter_info.xy, camera.filter_info.xy, camera_sample.filter_sample);
        let film_position = vec2<f32>(pixel) + filter_offset + vec2<f32>(0.5);
        let sample_time = mix(camera.camera_info.z, camera.camera_info.w, camera_sample.time);
        _ = sample_time;
        accumulated += render_sample(film_position, camera_sample.lens, pixel, sample_index);
    }
    output[global_id.y * width + global_id.x] = vec4<f32>(
        accumulated / f32(camera.sampler_info.w),
        1.0,
    );
}