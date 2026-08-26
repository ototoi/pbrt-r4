@compute @workgroup_size(64)
fn handle_emissive_intersection(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot_index = global_id.x;
    if (slot_index >= pixel_count() || slot_index >= arena.capacity) {
        return;
    }
    let ray = arena.rays[slot_index];
    if (ray.indices.y != RAY_STATE_SURFACE) {
        return;
    }
    let primitive_index = ray.indices.z;
    let triangle_index = ray.indices.w;
    let barycentrics = vec3<f32>(
        1.0 - ray.hit.y - ray.hit.z,
        ray.hit.y,
        ray.hit.z,
    );
    var found = false;
    var emission = vec3<f32>(0.0);
    for (var light_index = 0u; light_index < light_source_count(); light_index += 1u) {
        let source = lights[light_index];
        if (source.kind != 2u) {
            continue;
        }
        for (var geometry_index = 0u; geometry_index < source.triangle; geometry_index += 1u) {
            let geometry = area_light_geometry(source, geometry_index);
            if (geometry.primitive == primitive_index && geometry.triangle == triangle_index) {
                let triangle = area_light_triangle(geometry);
                var normal = area_light_normal(geometry, barycentrics, triangle);
                let cosine = dot(normal, -ray.direction.xyz);
                if ((source.flags & 1u) != 0u) {
                    normal = select(normal, -normal, cosine < 0.0);
                }
                if ((source.flags & 1u) != 0u || cosine > 0.0) {
                    emission += source.intensity.xyz;
                    found = true;
                }
            }
        }
    }
    if (found) {
        arena.rays[slot_index].radiance +=
            vec4<f32>(ray.throughput.xyz * emission, 0.0);
        arena.rays[slot_index].indices.y = RAY_STATE_VISIBLE;
    }
}
