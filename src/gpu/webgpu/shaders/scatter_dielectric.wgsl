@compute @workgroup_size(64, 1, 1)
fn scatter_dielectric(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let queue_index = global_id.y * INDIRECT_ROW_ITEMS + global_id.x;
    if (queue_index >= scatter_dielectric_count()) {
        return;
    }
    let ray_index = load_scatter_dielectric_ray(queue_index);
    let ray = load_current_ray(ray_index);
    let pixel_index = ray.pixel_index;
    let surface = surfaces[pixel_index];
    let evaluated = resolve_attributes_eval_work_item(surface.attributes_eval_work_item);
    let eta_attribute = load_material_attribute(evaluated.material_node, 0u);
    if (eta_attribute.kind != 1u) { terminate_secondary_wavelengths(pixel_index); }
    var eta = evaluated.values[0].x;
    if (eta == 0.0) { eta = 1.0; }
    if (!(eta > 0.0) || eta != eta) {
        set_render_error();
        return;
    }
    let normal = normalize(surface.normal.xyz);
    let tangent = surface.tangent.xyz;
    let wo = scattering_local_frame(normalize(-ray.direction.xyz), tangent, normal);
    if (wo.z == 0.0) { return; }

    // Direct lighting. pbrt-v4 DielectricBxDF::f/PDF are non-zero only for a
    // rough interface (classify_surface_scatter only enqueues rough
    // dielectrics into the direct-lighting queue).
    let light_sample = direct_light_samples[pixel_index];
    if (light_sample.valid != 0u) {
        let wi = scattering_local_frame(light_sample.direction_pdf.xyz, tangent, normal);
        let bsdf_pdf = dielectric_interface_pdf(evaluated, wo, wi, true, true);
        if (bsdf_pdf > 0.0) {
            let f = dielectric_interface_f(evaluated, wo, wi);
            add_direct_lighting(ray, surface, light_sample, f, bsdf_pdf, abs(wi.z));
        }
    }

    // Indirect bounce.
    let samples = load_ray_samples(pixel_index);
    let bs = sample_dielectric_interface(
        evaluated, wo, samples.indirect.x, samples.indirect.yz, true, true,
    );
    if (bs.valid == 0u || bs.pdf <= 0.0 || bs.wi.z == 0.0) { return; }
    let direction = normalize(scattering_world_frame(bs.wi, tangent, normal));
    let next_throughput = ray.throughput * bs.f * abs(bs.wi.z) / bs.pdf;
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
        ray.inv_w_u / bs.pdf,
        bs.pdf,
        bs.specular, 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&queue_counters.next.overflow, 1u);
        return;
    }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
