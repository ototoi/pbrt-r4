@compute @workgroup_size(8, 8, 1)
fn accumulate_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.tile_width || global_id.y >= viewport.tile_height) {
        return;
    }
    let pixel_index = global_id.y * viewport.tile_width + global_id.x;
    // The framebuffer is sized to the rendered region, not the tile, so its
    // address is the tile's absolute position with the region's offset
    // subtracted back out.
    let region_pixel = vec2<u32>(
        viewport.tile_x + global_id.x - viewport.region_x,
        viewport.tile_y + global_id.y - viewport.region_y,
    );
    let framebuffer_index = region_pixel.y * viewport.region_width + region_pixel.x;
    if (film_params.mode != 0u) {
        framebuffer[framebuffer_index] += vec4<f32>(load_sample_radiance(pixel_index).xyz, 1.0);
        return;
    }
    let lambda = load_sample_lambda(pixel_index);
    let weighted = safe_div_spectrum(load_sample_radiance(pixel_index), load_sample_lambda_pdf(pixel_index));
    var sensor_rgb = vec3<f32>(
        average_spectrum(weighted * evaluate_spectrum(film_params.sensor_response.x, lambda)),
        average_spectrum(weighted * evaluate_spectrum(film_params.sensor_response.y, lambda)),
        average_spectrum(weighted * evaluate_spectrum(film_params.sensor_response.z, lambda)),
    ) * film_params.imaging_ratio;
    let maximum = max(sensor_rgb.x, max(sensor_rgb.y, sensor_rgb.z));
    if (film_params.max_sample_luminance > 0.0 && maximum > film_params.max_sample_luminance) {
        sensor_rgb *= film_params.max_sample_luminance / maximum;
    }
    framebuffer[pixel_index] += vec4<f32>(sensor_rgb, 1.0);
}
