@compute @workgroup_size(64)
fn handle_escaped_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot_index = global_id.x;
    if (slot_index >= pixel_count() || slot_index >= arena.capacity) {
        return;
    }
    let ray = arena.rays[slot_index];
    if (ray.indices.y != RAY_STATE_MISS) {
        return;
    }

    let source_count = min(lights[0].flags >> 16u, arrayLength(&lights));
    var radiance = vec3<f32>(0.0);
    for (var light_index = 0u; light_index < source_count; light_index += 1u) {
        let light = lights[light_index];
        if (light.kind == 1u) {
            radiance += light.intensity.xyz;
        }
    }
    arena.rays[slot_index].radiance +=
        vec4<f32>(ray.throughput.xyz * radiance, 0.0);
    arena.rays[slot_index].indices.y = RAY_STATE_VISIBLE;
}
