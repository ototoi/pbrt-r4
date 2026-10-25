@compute @workgroup_size(64, 1, 1)
fn scatter_hair(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= scatter_hair_count()) { return; }
    let ray_index = load_scatter_hair_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);

    // pbrt-v4 BSDF shading frame: Frame::FromXZ(Normalize(dpdu), ns), so the
    // hair fiber runs along local x (sinθ) and the shading normal is local z.
    // On diced curves surface.tangent falls back to the UV-affine dpdu, which
    // follows the fiber because dice UVs run u along the curve.
    let normal = normalize(surface.normal.xyz);
    let tangent = normalize(surface.tangent.xyz);
    // HairMaterial::GetBxDF: h = -1 + 2 * uv.y across the curve width.
    let h = -1.0 + 2.0 * surface.uv.y;
    let bxdf = hair_bxdf_make(evaluated, h);

    let wo = -ray.direction.xyz;
    let wo_local = frame_to_local(wo, tangent, normal);

    // Direct lighting. hair is never EffectivelySmooth, so
    // classify_surface_scatter enqueues direct_eval for every hair ray.
    let light_sample = direct_light_samples[pixel_index];
    if (light_sample.valid != 0u) {
        let wi = light_sample.direction_pdf.xyz;
        let wi_local = frame_to_local(wi, tangent, normal);
        let cosine = abs(dot(normal, wi));
        if (cosine > 0.0) {
            // HairBxDF is two-sided (f divides by AbsCosTheta), so light
            // directions on either side of the shading frame contribute.
            let f = hair_bxdf_f(bxdf, wo_local, wi_local);
            let bsdf_pdf = hair_bxdf_pdf(bxdf, wo_local, wi_local);
            add_direct_lighting(ray, surface, light_sample, f, bsdf_pdf, cosine);
        }
    }

    // Indirect bounce.
    let samples = load_ray_samples(pixel_index);
    let sampled = hair_bxdf_sample_f(bxdf, wo_local, samples.indirect.xy);
    if (sampled.valid == 0u || sampled.pdf <= 0.0 || all(sampled.f == vec4<f32>(0.0))) {
        return;
    }
    let direction = normalize(frame_from_local(sampled.wi, tangent, normal));
    var next_beta = ray.beta * sampled.f * abs(sampled.wi.z) / sampled.pdf;
    if (ray.depth >= 1u) {
        let rr_beta = max(
            sampled_spectrum_max_component_value(next_beta * ray.eta_scale),
            0.0,
        ) / max(sampled_spectrum_average(ray.r_u), 1e-7);
        let q = max(0.0, 1.0 - rr_beta);
        if (samples.indirect.w < q) { return; }
        next_beta /= max(1.0 - q, 1e-7);
    }
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz,
            surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0),
        next_beta, ray.r_u, ray.r_u / sampled.pdf,
        surface.position, surface.position_error, surface.geometric_normal,
        vec4<f32>(normal, 0.0), pixel_index, ray.depth + 1u,
        ray.eta_scale, sampled.pdf, 0u, interaction_get_medium(ray, surface, direction), 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&queue_counters.next.overflow, 1u);
        return;
    }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
