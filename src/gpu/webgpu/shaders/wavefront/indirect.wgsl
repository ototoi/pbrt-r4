fn cosine_sample(u: vec2<f32>) -> vec3<f32> {
    let r = sqrt(u.x);
    let phi = 6.283185307179586 * u.y;
    return vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

@compute @workgroup_size(64)
fn sample_indirect_bxdf(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot_index = global_id.x;
    if (slot_index >= pixel_count() || slot_index >= arena.capacity) {
        return;
    }
    let ray = arena.rays[slot_index];
    if (ray.indices.y != RAY_STATE_BOUNCE) {
        return;
    }
    if (ray.direct_lighting.w > 0.5) {
        arena.rays[slot_index].radiance += vec4<f32>(ray.direct_lighting.xyz, 0.0);
    }
    let depth = ray.hit.w;
    if (depth + 1.0 >= f32(camera.bvh_info.z)) {
        arena.rays[slot_index].indices.y = RAY_STATE_VISIBLE;
        return;
    }

    let pixel = vec2<i32>(
        i32(ray.indices.x % u32(camera.viewport.z)) + i32(camera.viewport.x),
        i32(ray.indices.x / u32(camera.viewport.z)) + i32(camera.viewport.y),
    );
    let sample = independent_ray_sample(pixel, arena.sample_index, u32(depth)).indirect;
    let n = normalize(ray.surface_normal.xyz);
    let tangent = normalize(select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.x) > 0.9));
    let bitangent = cross(n, tangent);
    let local = cosine_sample(sample.direction);
    let direction = normalize(tangent * local.x + bitangent * local.y + n * local.z);
    arena.rays[slot_index].origin = vec4<f32>(
        offset_ray_origin(ray.surface_position.xyz, ray.surface_error.xyz, n, direction),
        1.0,
    );
    arena.rays[slot_index].direction = vec4<f32>(direction, 0.0);
    arena.rays[slot_index].throughput *= ray.material_reflectance;
    arena.rays[slot_index].hit = vec4<f32>(0.0, 0.0, 0.0, depth + 1.0);
    arena.rays[slot_index].indices.y = RAY_STATE_ACTIVE;
}
