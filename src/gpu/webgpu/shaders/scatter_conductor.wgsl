@compute @workgroup_size(64, 1, 1)
fn scatter_conductor(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= scatter_conductor_count()) { return; }
    let ray_index = load_scatter_conductor_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);
    let roughness = evaluated.values[2].x;

    // Direct lighting. classify_surface_scatter only enqueues this ray into
    // direct_eval when roughness >= 1e-3 (glossy); below that, a smooth
    // conductor is Specular in pbrt-v4 and direct_light_samples[pixel_index]
    // may hold a stale entry from an earlier depth, so gate on the same
    // predicate rather than trusting `valid` alone.
    let light_sample = direct_light_samples[pixel_index];
    if (roughness >= 1e-3 && light_sample.valid != 0u) {
        let wo = -ray.direction.xyz;
        let wi = light_sample.direction_pdf.xyz;
        let shading_n = surface.normal.xyz;
        let cos_wo = dot(shading_n, wo);
        let cos_wi = dot(shading_n, wi);
        if (cos_wo * cos_wi > 0.0) {
            let cosine = abs(cos_wi);
            if (cosine > 0.0) {
                let eta = evaluated.values[0];
                let k = evaluated.values[1];
                let h = scattering_local(normalize(wo + wi), shading_n);
                let fresnel = conductor_fresnel(dot(scattering_local(wo, shading_n), h), eta, k);
                let alpha = max(evaluated.values[2].x, 1e-3);
                let cos_h = max(abs(h.z), 1e-5);
                let alpha2 = alpha * alpha;
                let d = alpha2 / (PI * pow(cos_h * cos_h * (alpha2 - 1.0) + 1.0, 2.0));
                let cos_o = max(abs(cos_wo), 1e-5);
                let cos_i = max(abs(cos_wi), 1e-5);
                let g_o = 2.0 * cos_o / (cos_o + sqrt(cos_o * cos_o + alpha2 * (1.0 - cos_o * cos_o)));
                let g_i = 2.0 * cos_i / (cos_i + sqrt(cos_i * cos_i + alpha2 * (1.0 - cos_i * cos_i)));
                let f = fresnel * d * g_o * g_i / (4.0 * cos_o * cos_i);
                let bsdf_pdf = d * cos_h / max(4.0 * abs(dot(scattering_local(wo, shading_n), h)), 1e-5);
                add_direct_lighting(ray, surface, light_sample, f, bsdf_pdf, cosine);
            }
        }
    }

    // Indirect bounce.
    let lambda = load_sample_lambda(pixel_index);
    let normal = normalize(surface.normal.xyz);
    let wo = normalize(-ray.direction.xyz);
    let tangent = make_tangent(normal);
    let bitangent = cross(normal, tangent);
    var half_local = vec3<f32>(0.0, 0.0, 1.0);
    var pdf = 1.0;
    if (roughness >= 1e-3) {
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
    if (roughness >= 1e-3) {
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
    var next_throughput = ray.throughput * f * cos_i / pdf;
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz,
            surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0),
        next_throughput,
        surface.position, surface.position_error, surface.geometric_normal,
        vec4<f32>(normal, 0.0), pixel_index, ray.depth + 1u,
        ray.inv_w_u, ray.inv_w_u / pdf, pdf, u32(roughness < 1e-3), 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&queue_counters.next.overflow, 1u);
        return;
    }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
