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
        arena.rays[slot_index].direct_lighting.w = 0.0;
        arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
        return;
    }
    let light_index = min(
        u32(murmur_hash_float_ray(ray.origin.xyz, ray.direction.xyz) * f32(source_count)),
        source_count - 1u,
    );
    let light = lights[light_index];
    if (light.kind == 1u) {
        arena.rays[slot_index].direct_lighting.w = 0.0;
        arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
        return;
    }
    var light_position = light.position.xyz;
    var emitted_cosine = 1.0;
    var pdf_area = 1.0;
    if (light.kind == 2u) {
        if (light.triangle == 0u) {
            arena.rays[slot_index].direct_lighting.w = 0.0;
            arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
            return;
        }
        let pixel = vec2<i32>(
            i32(ray.indices.x % u32(camera.viewport.z)) + i32(camera.viewport.x),
            i32(ray.indices.x / u32(camera.viewport.z)) + i32(camera.viewport.y),
        );
        let sample = independent_ray_sample(pixel, arena.sample_index, u32(ray.hit.w)).direct;
        let geometry_index = min(
            u32(sample.light_selection * f32(light.triangle)),
            light.triangle - 1u,
        );
        let geometry = area_light_geometry(light, geometry_index);
        let triangle = area_light_triangle(geometry);
        let root_u = sqrt(sample.light_sample.x);
        let barycentrics = vec3<f32>(
            1.0 - root_u,
            root_u * (1.0 - sample.light_sample.y),
            root_u * sample.light_sample.y,
        );
        light_position = triangle[0] * barycentrics.x
            + triangle[1] * barycentrics.y
            + triangle[2] * barycentrics.z;
        var light_normal = normalize(cross(
            triangle[1] - triangle[0],
            triangle[2] - triangle[0],
        ));
        let geometry_primitive = primitives[geometry.primitive];
        if ((geometry_primitive.flags & 1u) != 0u) {
            light_normal = -light_normal;
        }
        let area = 0.5 * length(cross(
            triangle[1] - triangle[0],
            triangle[2] - triangle[0],
        ));
        if (area <= 0.0) {
            arena.rays[slot_index].direct_lighting.w = 0.0;
            arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
            return;
        }
        pdf_area = 1.0 / (f32(light.triangle) * area);
        // A two-sided emitter contributes from either side; a one-sided emitter
        // contributes only when its geometric normal faces the shading point.
        emitted_cosine = dot(light_normal, ray.surface_position.xyz - light_position);
        if ((light.flags & 1u) != 0u) {
            emitted_cosine = abs(emitted_cosine);
        }
        if (emitted_cosine <= 0.0) {
            arena.rays[slot_index].direct_lighting.w = 0.0;
            arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
            return;
        }
    } else if (light.kind != 0u) {
        arena.rays[slot_index].direct_lighting.w = 0.0;
        arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
        return;
    }
    let to_light = light_position - ray.surface_position.xyz;
    let distance_squared = dot(to_light, to_light);
    if (distance_squared == 0.0) {
        arena.rays[slot_index].direct_lighting.w = 0.0;
        arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
        return;
    }
    let distance = sqrt(distance_squared);
    let direction = to_light / distance;
    let cosine = dot(ray.surface_normal.xyz, direction);
    if (cosine <= 0.0) {
        arena.rays[slot_index].direct_lighting.w = 0.0;
        arena.rays[slot_index].indices.y = RAY_STATE_BOUNCE;
        return;
    }

    let radiance = light.intensity.xyz
        * (cosine * emitted_cosine * f32(source_count) * INV_PI
            / (distance_squared * pdf_area));
    arena.rays[slot_index].direct_lighting = vec4<f32>(
        ray.throughput.xyz * ray.material_reflectance.xyz * radiance,
        1.0,
    );
    let origin = offset_ray_origin(
        ray.surface_position.xyz,
        ray.surface_error.xyz,
        ray.surface_normal.xyz,
        to_light,
    );
    arena.rays[slot_index].origin = vec4<f32>(origin, 1.0);
    arena.rays[slot_index].direction = vec4<f32>(light_position - origin, 0.0);
    arena.rays[slot_index].hit.x = 1.0 - SHADOW_EPSILON;
    arena.rays[slot_index].indices.y = RAY_STATE_SHADOW;
}
