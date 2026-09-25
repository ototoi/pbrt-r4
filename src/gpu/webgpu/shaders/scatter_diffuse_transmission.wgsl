@compute @workgroup_size(64, 1, 1)
fn scatter_diffuse_transmission(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= scatter_diffuse_transmission_count()) { return; }
    let ray_index = load_scatter_diffuse_transmission_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);

    // Direct lighting.
    let light_sample = direct_light_samples[pixel_index];
    if (light_sample.valid != 0u) {
        let reflectance = evaluated.values[0];
        let transmittance = evaluated.values[1];
        let wo = -ray.direction.xyz;
        let wi = light_sample.direction_pdf.xyz;
        let shading_n = surface.normal.xyz;
        let cos_wo = dot(shading_n, wo);
        let cos_wi = dot(shading_n, wi);
        let cosine = abs(cos_wi);
        if (cosine > 0.0) {
            let pr = max_spectrum(reflectance);
            let pt = max_spectrum(transmittance);
            let total = pr + pt;
            if (total > 0.0) {
                let same_side = cos_wo * cos_wi > 0.0;
                let branch_probability = select(pt / total, pr / total, same_side);
                let f = select(transmittance, reflectance, same_side) / PI;
                let bsdf_pdf = cosine / PI * branch_probability;
                add_direct_lighting(ray, surface, light_sample, f, bsdf_pdf, cosine);
            }
        }
    }

    // Indirect bounce.
    let r = evaluated.values[0];
    let t = evaluated.values[1];
    let pr = max_spectrum(r);
    let pt = max_spectrum(t);
    let total = pr + pt;
    if (total <= 0.0) { return; }

    let samples = load_ray_samples(pixel_index);
    let reflect = samples.direct.w < pr / total;
    let normal = surface.normal.xyz;
    let tangent = make_tangent(normal);
    let bitangent = cross(normal, tangent);
    let wo = -ray.direction.xyz;
    let u = vec2<f32>(samples.indirect.y, samples.indirect.z);
    let radius = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    var local = vec3<f32>(radius * cos(phi), radius * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
    let wo_side = dot(normal, wo) >= 0.0;
    if (reflect != wo_side) { local.z = -local.z; }
    let direction = normalize(tangent * local.x + bitangent * local.y + normal * local.z);
    let cosine = abs(dot(normal, direction));
    let branch_probability = select(pt / total, pr / total, reflect);
    let pdf = cosine / PI * branch_probability;
    if (pdf <= 0.0) { return; }
    let f = select(t, r, reflect) / PI;
    var next_throughput = ray.throughput * f * cosine / pdf;
    if (ray.depth >= 1u) {
        let rr_beta = max(max_spectrum(next_throughput), 0.0) / max(ray.inv_w_u, 1e-7);
        let q = max(0.0, 1.0 - rr_beta);
        if (samples.indirect.w < q) { return; }
        next_throughput /= max(1.0 - q, 1e-7);
    }
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz,
            surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0), next_throughput,
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
