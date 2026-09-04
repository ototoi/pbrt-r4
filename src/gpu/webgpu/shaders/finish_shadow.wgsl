@compute @workgroup_size(8, 8, 1)
fn finish_shadow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let index = global_id.y * viewport.width + global_id.x;
    if (index >= shadow_ray_count()) {
        return;
    }
    let pixel_index = load_shadow_pixel(index);
    if (surfaces[pixel_index].shadow_visible != 1u) {
        surfaces[pixel_index].direct = vec4<f32>(0.0);
    }
}
