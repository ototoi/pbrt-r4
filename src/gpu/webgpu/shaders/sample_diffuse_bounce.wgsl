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
    let samples = load_ray_samples(pixel_index);
    let surface = surfaces[pixel_index];
    if (surface.hit == 0u || surface.flags != 0u) {
        return;
    }
    let material_kind = load_material_kind(surface.material);
    if (material_kind != MATERIAL_KIND_DIFFUSE) {
        return;
    }
    let normal = surface.normal.xyz;
    let tangent = make_tangent(normal);
    let bitangent = cross(normal, tangent);
    let u = vec2<f32>(samples.indirect.y, samples.indirect.z);
    let radius = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    var local = vec3<f32>(
        radius * cos(phi),
        radius * sin(phi),
        sqrt(max(0.0, 1.0 - u.x)),
    );
    // pbrt-v4 DiffuseBxDF::Sample_f: sample the cosine hemisphere on the
    // side of the shading frame that contains wo (`if (wo.z < 0) wi.z *= -1`),
    // and use AbsCosTheta for the pdf. This bounces outward even when the
    // mesh shading normals are globally inverted.
    let wo = -ray.direction.xyz;
    if (dot(normal, wo) < 0.0) {
        local.z = -local.z;
    }
    let direction = normalize(tangent * local.x + bitangent * local.y + normal * local.z);
    let next_pdf = abs(dot(normal, direction)) / PI;
    var next_throughput = ray.throughput;
    if (ray.depth >= 1u) {
        let rr_beta = max(
            max(next_throughput.x, max(next_throughput.y, next_throughput.z)),
            0.0,
        ) / max(ray.inv_w_u, 1e-7);
        let q = max(0.0, 1.0 - rr_beta);
        if (samples.indirect.w < q) {
            return;
        }
        next_throughput = next_throughput / max(1.0 - q, 1e-7);
    }
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz, surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0),
        next_throughput,
        surface.position,
        surface.position_error,
        surface.geometric_normal,
        vec4<f32>(normal, 0.0),
        pixel_index,
        ray.depth + 1u,
        ray.inv_w_u,
        ray.inv_w_u / max(next_pdf, 1e-7),
        next_pdf,
        vec3<u32>(0u, 0u, 0u),
    );
    let next_index = atomicAdd(&wavefront_queue[NEXT_COUNT], 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&wavefront_queue[NEXT_OVERFLOW], 1u);
        return;
    }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
