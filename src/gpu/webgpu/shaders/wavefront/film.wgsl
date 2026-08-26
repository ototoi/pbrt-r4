@compute @workgroup_size(64)
fn update_film(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot_index = global_id.x;
    if (slot_index >= pixel_count() || slot_index >= arena.capacity) {
        return;
    }
    let ray = arena.rays[slot_index];
    if (ray.indices.y == RAY_STATE_VISIBLE) {
        output[ray.indices.x] += vec4<f32>(ray.radiance.xyz, 1.0);
        return;
    }
}
