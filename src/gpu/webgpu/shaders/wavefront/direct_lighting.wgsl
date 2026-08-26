const INV_PI: f32 = 0.3183098861837907;
const SHADOW_EPSILON: f32 = 0.0001;

@compute @workgroup_size(64)
fn sample_direct_lighting(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot_index = global_id.x;
    if (slot_index >= pixel_count() || slot_index >= arena.capacity) {
        return;
    }

    let ray = arena.rays[slot_index];
    if (ray.indices.y != RAY_STATE_MATERIAL) {
        return;
    }

    let source_count = min(lights[0].flags >> 16u, arrayLength(&lights));
    if (source_count == 0u) {
        arena.rays[slot_index].indices.y = RAY_STATE_OCCLUDED;
        return;
    }
    let light_index = min(
        u32(murmur_hash_float_ray(ray.origin.xyz, ray.direction.xyz) * f32(source_count)),
        source_count - 1u,
    );
    let light = lights[light_index];
    if (light.kind != 0u) {
        arena.rays[slot_index].indices.y = RAY_STATE_OCCLUDED;
        return;
    }
    let to_light = light.position.xyz - ray.surface_position.xyz;
    let distance_squared = dot(to_light, to_light);
    if (distance_squared == 0.0) {
        arena.rays[slot_index].indices.y = RAY_STATE_OCCLUDED;
        return;
    }
    let distance = sqrt(distance_squared);
    let direction = to_light / distance;
    let cosine = dot(ray.surface_normal.xyz, direction);
    if (cosine <= 0.0) {
        arena.rays[slot_index].indices.y = RAY_STATE_OCCLUDED;
        return;
    }

    let radiance = light.intensity.xyz
        * (cosine * f32(source_count) * INV_PI / distance_squared);
    arena.rays[slot_index].direct_lighting = vec4<f32>(
        ray.material_reflectance.xyz * radiance,
        0.0,
    );
    let origin = offset_ray_origin(
        ray.surface_position.xyz,
        ray.surface_error.xyz,
        ray.surface_normal.xyz,
        to_light,
    );
    arena.rays[slot_index].origin = vec4<f32>(origin, 1.0);
    arena.rays[slot_index].direction = vec4<f32>(light.position.xyz - origin, 0.0);
    arena.rays[slot_index].hit.x = 1.0 - SHADOW_EPSILON;
    arena.rays[slot_index].indices.y = RAY_STATE_SHADOW;
}
