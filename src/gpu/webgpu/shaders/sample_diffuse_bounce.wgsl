@compute @workgroup_size(8, 8, 1)
fn sample_diffuse_bounce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= current_ray_count()) {
        return;
    }
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    if (surface.hit == 0u || surface.flags != 0u) {
        return;
    }
    let material_kind = load_material_kind(surface.material);
    if (material_kind != MATERIAL_KIND_DIFFUSE) {
        return;
    }
    store_sample_radiance(
        pixel_index,
        vec4<f32>(load_sample_radiance(pixel_index).xyz + ray.throughput.xyz * surface.direct.xyz, 1.0),
    );

    let normal = surface.normal.xyz;
    let tangent = make_tangent(normal);
    let bitangent = cross(normal, tangent);
    let u = vec2<f32>(random01(pixel_index, 3u), random01(pixel_index, 4u));
    let radius = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let local = vec3<f32>(
        radius * cos(phi),
        radius * sin(phi),
        sqrt(max(0.0, 1.0 - u.x)),
    );
    let direction = normalize(tangent * local.x + bitangent * local.y + normal * local.z);
    let next_pdf = max(dot(normal, direction), 0.0) / PI;
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz, normal), 1.0),
        vec4<f32>(direction, 0.0),
        ray.throughput * vec4<f32>(0.5, 0.5, 0.5, 0.0),
        pixel_index,
        ray.depth + 1u,
        0u,
        next_pdf,
    );
    let next_index = atomicAdd(&wavefront_queue[NEXT_COUNT], 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&wavefront_queue[NEXT_OVERFLOW], 1u);
        return;
    }
    store_next_ray(next_index, next_ray);
}
