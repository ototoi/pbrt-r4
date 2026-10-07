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
    if (!material_spectrum_attribute_is_constant(evaluated.material_node, 0u)) {
        terminate_secondary_wavelengths(pixel_index);
    }
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
    var next_beta = ray.beta * bs.f * abs(bs.wi.z) / bs.pdf;
    var eta_scale = ray.eta_scale;
    if (bs.transmission != 0u) {
        eta_scale = eta_scale * bs.etap * bs.etap;
    }
    if (evaluated.bxdf_kind == MATERIAL_KIND_SUBSURFACE) {
        if (bs.transmission != 0u) {
            let material = bssrdf_materials[material_nodes[evaluated.material_node].bssrdf_index];
            var sigma_a = max(vec4<f32>(0.0), material.scale * evaluated.values[4]);
            var sigma_s = max(vec4<f32>(0.0), material.scale * evaluated.values[5]);
            if (material.coefficient_kind == 1u) {
                let reflectance = clamp(evaluated.values[4], vec4<f32>(0.0), vec4<f32>(1.0));
                let mfp = max(vec4<f32>(1e-6), material.scale * evaluated.values[5]);
                for (var i = 0u; i < 4u; i++) {
                    let rho = bssrdf_invert_reflectance(bssrdf_tables[material.table_index], reflectance[i]);
                    sigma_s[i] = rho / mfp[i];
                    sigma_a[i] = (1.0 - rho) / mfp[i];
                }
            }
            let sigma_t = sigma_a + sigma_s;
            var rho = vec4<f32>(0.0);
            for (var i = 0u; i < 4u; i++) {
                if (sigma_t[i] != 0.0) { rho[i] = sigma_s[i] / sigma_t[i]; }
            }
            if (any(sigma_t > vec4<f32>(0.0))) {
                let dimension = 6u + 13u * ray.depth + 8u;
                let sample = vec4<f32>(sampler_get_1d(pixel_index, dimension),
                    sampler_get_2d(pixel_index, dimension + 1u), material.eta);
                let index = atomicAdd(&bssrdf_work.state.count, 1u);
                if (index >= bssrdf_work.state.capacity) {
                    atomicStore(&bssrdf_work.state.overflow, 1u); set_render_error(); return;
                }
                bssrdf_work.items[index] = BSSRDFProbeWorkItem(surface.position,
                    vec4<f32>(normal, 0.0), sigma_t, rho, sample, material.table_index,
                    surface.material_root, pixel_index, ray_index);
                var entry_ray = ray;
                entry_ray.beta = next_beta;
                entry_ray.eta_scale = eta_scale;
                store_current_ray(ray_index, entry_ray);
                return;
            }
        }
        let average_r_u = dot(ray.r_u, vec4<f32>(0.25));
        let rr_beta = next_beta * eta_scale / average_r_u;
        let rr_max = max(max(rr_beta.x, rr_beta.y), max(rr_beta.z, rr_beta.w));
        if (ray.depth >= 1u && rr_max < 1.0) {
            let q = max(0.0, 1.0 - rr_max);
            if (samples.indirect.w < q) { return; }
            next_beta /= 1.0 - q;
        }
    }
    let next_ray = RayWorkItem(
        vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz, surface.geometric_normal.xyz, direction), 1.0),
        vec4<f32>(direction, 0.0),
        next_beta,
        ray.r_u,
        ray.r_u / bs.pdf,
        surface.position,
        surface.position_error,
        surface.geometric_normal,
        vec4<f32>(normal, 0.0),
        pixel_index,
        ray.depth + 1u,
        eta_scale,
        bs.pdf,
        bs.specular, medium_for_direction(ray, surface, direction), 0u, 0u,
    );
    let next_index = atomicAdd(&queue_counters.next.count, 1u);
    if (next_index >= pixel_count()) {
        atomicStore(&queue_counters.next.overflow, 1u);
        return;
    }
    store_ray_samples(pixel_index, generate_ray_samples(pixel_index, ray.depth + 1u));
    store_next_ray(next_index, next_ray);
}
