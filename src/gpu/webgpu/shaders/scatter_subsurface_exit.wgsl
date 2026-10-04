@compute @workgroup_size(64)
fn scatter_subsurface_exit(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.y * INDIRECT_ROW_ITEMS + id.x;
    if (index >= atomicLoad(&bssrdf_work.state.count)) { return; }
    if (bssrdf_results[index].valid == 0u) { return; }
    let work = bssrdf_work.items[index];
    let ray = load_current_ray(work.ray_index);
    let surface = surfaces[work.pixel_index];
    let normal = surface.normal.xyz;
    let tangent = surface.tangent.xyz;
    let eta = work.sample.w;
    let light_sample = direct_light_samples[work.pixel_index];
    if (light_sample.valid != 0u) {
        let cosine = dot(light_sample.direction_pdf.xyz, normal);
        if (cosine > 0.0) {
            add_direct_lighting(ray, surface, light_sample,
                bssrdf_normalized_fresnel(cosine, eta), cosine / PI, cosine);
        }
    }
    let samples = load_ray_samples(work.pixel_index);
    let wi = bssrdf_sample_cosine(samples.indirect.yz);
    let pdf = wi.z / PI;
    if (pdf <= 0.0) { return; }
    var beta = ray.beta * bssrdf_normalized_fresnel(wi.z, eta) * wi.z / pdf;
    let rr_beta = beta * ray.eta_scale / dot(ray.r_u, vec4<f32>(0.25));
    let rr_max = max(max(rr_beta.x, rr_beta.y), max(rr_beta.z, rr_beta.w));
    if (ray.depth > 1u && rr_max < 1.0) {
        let q = max(0.0, 1.0 - rr_max);
        if (samples.indirect.w < q) { return; }
        beta /= 1.0 - q;
    }
    if (!any(beta > vec4<f32>(0.0))) { return; }
    let direction = normalize(scattering_world_frame(wi, tangent, normal));
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz,
            surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0), beta, ray.r_u, ray.r_u / pdf,
        surface.position, surface.position_error, surface.geometric_normal, surface.normal,
        work.pixel_index, ray.depth + 1u, ray.eta_scale, pdf, 0u,
        medium_for_direction(ray, surface, direction), 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) { atomicStore(&queue_counters.next.overflow, 1u); return; }
    store_next_ray(next_index, next_ray);
    store_ray_samples(work.pixel_index, generate_ray_samples(work.pixel_index, ray.depth + 1u));
}
