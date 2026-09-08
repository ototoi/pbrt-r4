@compute @workgroup_size(8, 8, 1)
fn sample_conductor_bounce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) { return; }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= current_ray_count()) { return; }
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let material_index = resolve_material_leaf(surface.material);
    if (surface.hit == 0u || surface.flags != 0u
        || load_material_kind(material_index) != MATERIAL_KIND_CONDUCTOR) { return; }
    let lambda = load_sample_lambda(pixel_index);
    let evaluated = load_evaluated_attributes(surface.evaluated_attributes_root);
    let roughness = evaluated.values[2].x;
    let normal = normalize(surface.normal.xyz);
    let wo = normalize(-ray.direction.xyz);
    let tangent = make_tangent(normal);
    let bitangent = cross(normal, tangent);
    var half_local = vec3<f32>(0.0, 0.0, 1.0);
    var pdf = 1.0;
    if (roughness > 1e-3) {
        let samples = load_ray_samples(pixel_index);
        let alpha = max(roughness, 1e-3);
        let tan2 = alpha * alpha * samples.indirect.x / max(1.0 - samples.indirect.x, 1e-7);
        let phi = 2.0 * PI * samples.indirect.y;
        let cos_theta = 1.0 / sqrt(1.0 + tan2);
        half_local = normalize(vec3<f32>(sqrt(max(0.0, 1.0 - cos_theta * cos_theta)) * cos(phi),
            sqrt(max(0.0, 1.0 - cos_theta * cos_theta)) * sin(phi), cos_theta));
        let wo_local = scattering_local(wo, normal);
        if (dot(wo_local, half_local) < 0.0) { half_local = -half_local; }
        let d = alpha * alpha / (PI * pow(half_local.z * half_local.z * (alpha * alpha - 1.0) + 1.0, 2.0));
        pdf = d * abs(half_local.z) / max(4.0 * abs(dot(wo_local, half_local)), 1e-5);
    }
    let half_world = normalize(tangent * half_local.x + bitangent * half_local.y + normal * half_local.z);
    let direction = normalize(reflect(-wo, half_world));
    let cos_i = abs(dot(direction, normal));
    if (cos_i <= 1e-5 || pdf <= 1e-7) { return; }
    var f = conductor_fresnel(cos_i, evaluated.values[0], evaluated.values[1]) / cos_i;
    if (roughness > 1e-3) {
        let wi_local = scattering_local(direction, normal);
        let wo_local = scattering_local(wo, normal);
        let h_local = scattering_local(normalize(wo + direction), normal);
        let alpha = max(roughness, 1e-3);
        let a2 = alpha * alpha;
        let d = a2 / (PI * pow(h_local.z * h_local.z * (a2 - 1.0) + 1.0, 2.0));
        let g = 2.0 * abs(wo_local.z) / (abs(wo_local.z) + sqrt(wo_local.z * wo_local.z + a2 * (1.0 - wo_local.z * wo_local.z)))
            * 2.0 * abs(wi_local.z) / (abs(wi_local.z) + sqrt(wi_local.z * wi_local.z + a2 * (1.0 - wi_local.z * wi_local.z)));
        f = conductor_fresnel(abs(dot(wo_local, h_local)), evaluated.values[0], evaluated.values[1]) * d * g
            / max(4.0 * abs(wo_local.z * wi_local.z), 1e-5);
    }
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz,
            surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0),
        ray.throughput * f * cos_i / pdf,
        surface.position, surface.position_error, surface.geometric_normal,
        vec4<f32>(normal, 0.0), pixel_index, ray.depth + 1u,
        ray.inv_w_u, ray.inv_w_u / pdf, pdf, 0u, 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&queue_counters.next.overflow, 1u);
        return;
    }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
