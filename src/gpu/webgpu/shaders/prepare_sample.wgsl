@compute @workgroup_size(8, 8, 1)
fn prepare_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let pixel_index = global_id.y * viewport.width + global_id.x;
    let second_queue_index = pixel_index + pixel_count();
    framebuffer[pixel_index] = vec4<f32>(0.0);
    rays[pixel_index].is_active = 0u;
    rays[second_queue_index].is_active = 0u;
    surfaces[pixel_index].hit = 0u;
    surfaces[pixel_index].shadow_visible = 0u;
    surfaces[pixel_index].flags = 0u;
    surfaces[pixel_index].direct = vec4<f32>(0.0);
}
