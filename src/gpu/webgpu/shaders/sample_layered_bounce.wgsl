@compute @workgroup_size(8, 8, 1)
fn sample_layered_bounce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) { return; }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= current_ray_count()) { return; }
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    if (surface.hit == 0u || surface.flags != 0u
        || load_material_kind(surface.material) != MATERIAL_KIND_LAYERED) { return; }
    let data = load_layered_bxdf(surface.material);
    let eta_node = load_layered_eta_node(surface.material);
    if (!dielectric_eta_is_constant(eta_node)) { terminate_secondary_wavelengths(pixel_index); }
    var eta = load_dielectric_eta(eta_node, load_sample_lambda(pixel_index)).x;
    if (eta == 0.0) { eta = 1.0; }
    let reflectance = load_layered_bottom_reflectance(surface.material, load_sample_lambda(pixel_index));
    let samples = load_ray_samples(pixel_index);
    let normal = normalize(surface.normal.xyz);
    let wo = scattering_local(-ray.direction.xyz, normal);
    let bs = layered_sample(data, eta, reflectance, wo, samples.indirect.x, samples.indirect.yz,
        layered_path_seed(pixel_index, ray.depth));
    if (bs.valid == 0u || bs.pdf == 0.0 || layered_max(bs.f) == 0.0) { return; }
    var beta = ray.throughput * bs.f * abs(bs.wi.z) / bs.pdf;
    if (ray.depth >= 1u) {
        let rr_beta = layered_max(beta) / ray.inv_w_u;
        if (rr_beta < 1.0) {
            let q = max(0.0, 1.0 - rr_beta);
            if (samples.indirect.w < q) { return; }
            beta /= 1.0 - q;
        }
    }
    let tangent = make_tangent(normal);
    let direction = normalize(tangent * bs.wi.x + cross(normal, tangent) * bs.wi.y + normal * bs.wi.z);
    // Sample_f's path PDF updates beta. PDF() supplies the MIS density.
    // Zero prev_pdf denotes a delta event: emissive-hit MIS must give weight 1.
    let pdf = layered_pdf(data, eta, wo, bs.wi.xyz);
    let prev_pdf = select(pdf, 0.0, (bs.flags & SCATTER_SPECULAR) != 0u);
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz,
            surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0), beta,
        surface.position, surface.position_error, surface.geometric_normal,
        vec4<f32>(normal, 0.0), pixel_index, ray.depth + 1u,
        ray.inv_w_u, ray.inv_w_u / pdf, prev_pdf, vec3<u32>(0u),
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&queue_counters.next.overflow, 1u);
        return;
    }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
