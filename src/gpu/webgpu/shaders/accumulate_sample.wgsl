@compute @workgroup_size(8, 8, 1)
fn accumulate_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let pixel_index = global_id.y * viewport.width + global_id.x;
    let accumulation_index = pixel_index + pixel_count();
    framebuffer[accumulation_index] = vec4<f32>(
        framebuffer[accumulation_index].xyz + load_sample_radiance(pixel_index).xyz,
        1.0,
    );
}
