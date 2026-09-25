@compute @workgroup_size(8, 8, 1)
fn scatter_measured(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) {
        return;
    }
    let queue_index = global_id.y * viewport.width + global_id.x;
    if (queue_index >= scatter_measured_count()) {
        return;
    }
    let ray_index = load_scatter_measured_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);
    let material_node = evaluated.material_node;
    let lambda = load_sample_lambda(pixel_index);

    // Direct lighting.
    let light_sample = direct_light_samples[pixel_index];
    if (light_sample.valid != 0u) {
        let wo = -ray.direction.xyz;
        let wi = light_sample.direction_pdf.xyz;
        let shading_n = surface.normal.xyz;
        let cos_wo = dot(shading_n, wo);
        let cos_wi = dot(shading_n, wi);
        if (cos_wo * cos_wi > 0.0) {
            let cosine = abs(cos_wi);
            if (cosine > 0.0) {
                let id = measured_id(material_node);
                if (id != 0xffffffffu) {
                    let local_wo = scattering_local_frame(wo, surface.tangent.xyz, shading_n);
                    let local_wi = scattering_local_frame(wi, surface.tangent.xyz, shading_n);
                    let f = measured_f(id, local_wo, local_wi, lambda);
                    let bsdf_pdf = measured_pdf(id, local_wo, local_wi);
                    add_direct_lighting(ray, surface, light_sample, f, bsdf_pdf, cosine);
                }
            }
        }
    }

    // Indirect bounce.
    let normal = surface.normal.xyz;
    let tangent = surface.tangent.xyz;
    let wo = -ray.direction.xyz;
    let id = measured_id(material_node);
    if (id == 0xffffffffu) { return; }
    let samples = load_ray_samples(pixel_index);
    let sampled = measured_sample_f(
        id,
        scattering_local_frame(wo, tangent, normal),
        vec2<f32>(samples.indirect.y, samples.indirect.z),
        lambda,
    );
    if (sampled.valid == 0u || sampled.pdf <= 0.0 || all(sampled.f == vec4<f32>(0.0))) {
        return;
    }
    let direction = normalize(scattering_world_frame(sampled.wi, tangent, normal));
    var next_throughput = ray.throughput * sampled.f * abs(sampled.wi.z) / sampled.pdf;
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
        ray.inv_w_u, ray.inv_w_u / sampled.pdf, sampled.pdf, 0u, 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&queue_counters.next.overflow, 1u);
        return;
    }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
