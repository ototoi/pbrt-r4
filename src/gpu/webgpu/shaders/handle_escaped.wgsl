@compute @workgroup_size(8, 8, 1)
fn handle_escaped(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * viewport.width + global_id.x;
    if (index >= escaped_ray_count()) {
        return;
    }
    // No infinite light is lowered in the initial GPU scope. The queue is
    // nevertheless consumed explicitly so a miss is not silently discarded.
}
