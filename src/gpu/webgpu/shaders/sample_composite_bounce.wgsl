@compute @workgroup_size(8, 8, 1)
fn sample_composite_bounce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= viewport.width || global_id.y >= viewport.height) { return; }
    let ray_index = global_id.y * viewport.width + global_id.x;
    if (ray_index >= current_ray_count()) { return; }
    let ray = load_current_ray(ray_index);
    let surface = surfaces[ray.pixel_index];
    if (surface.hit == 0u || surface.flags != 0u) { return; }
    let kind = load_material_kind(surface.material);
    if (kind != MATERIAL_KIND_COATED_DIFFUSE && kind != MATERIAL_KIND_COATED_CONDUCTOR) { return; }
    var root = load_attributes_eval_work_item(surface.attributes_eval_work_item);
    if (kind == MATERIAL_KIND_COATED_DIFFUSE || kind == MATERIAL_KIND_COATED_CONDUCTOR) {
        let params = load_layered_params(root, kind);
        let top = load_attributes_eval_work_item(root.child_work_item0);
        let bottom = load_attributes_eval_work_item(root.child_work_item1);
        let normal = normalize(surface.normal.xyz);
        let tangent = make_tangent(normal);
        let bitangent = cross(normal, tangent);
        var wo = scattering_local(normalize(-ray.direction.xyz), normal);
        if (wo.z == 0.0) { return; }
        var flip_wi = false;
        if (wo.z < 0.0) {
            wo = -wo;
            flip_wi = true;
        }

        let samples = load_ray_samples(ray.pixel_index);
        var bs = sample_dielectric_interface(
            top, wo, samples.indirect.x, samples.indirect.yz, true, true,
        );
        if (bs.valid == 0u || bs.pdf == 0.0 || bs.wi.z == 0.0) { return; }

        var result_f = bs.f;
        var result_pdf = bs.pdf;
        var result_wi = bs.wi;
        var result_valid = bs.transmission == 0u;
        if (!result_valid) {
            var w = bs.wi;
            var path_f = bs.f * abs(bs.wi.z);
            var path_pdf = bs.pdf;
            var z = max(params.thickness, 1.17549435e-38);
            let max_depth = max(1u, u32(params.max_depth));
            for (var layer_depth = 0u; layer_depth < max_depth; layer_depth++) {
                let random_base = 16u + layer_depth * 4u;
                let rr_beta = max_spectrum(path_f) / max(path_pdf, 1e-30);
                if (layer_depth > 3u && rr_beta < 0.25) {
                    let q = max(0.0, 1.0 - rr_beta);
                    if (random01(ray.pixel_index, random_base, ray.depth) < q) { break; }
                    path_pdf *= 1.0 - q;
                }
                if (w.z == 0.0) { break; }

                if (max_spectrum(params.albedo) > 0.0) {
                    let dz = sample_layered_exponential(
                        random01(ray.pixel_index, random_base + 1u, ray.depth),
                        1.0 / abs(w.z),
                    );
                    let zp = select(z - dz, z + dz, w.z > 0.0);
                    if (zp == z) { break; }
                    if (zp > 0.0 && zp < params.thickness) {
                        let phase_wi = sample_hg_direction(
                            normalize(-w),
                            vec2<f32>(
                                random01(ray.pixel_index, random_base + 2u, ray.depth),
                                random01(ray.pixel_index, random_base + 3u, ray.depth),
                            ),
                            params.g,
                        );
                        let phase_p = hg_phase(dot(normalize(-w), phase_wi), params.g);
                        if (phase_p == 0.0 || phase_wi.z == 0.0) { break; }
                        path_f *= params.albedo * phase_p;
                        path_pdf *= phase_p;
                        w = phase_wi;
                        z = zp;
                        continue;
                    }
                    z = clamp(zp, 0.0, params.thickness);
                } else {
                    z = select(params.thickness, 0.0, z == params.thickness);
                    path_f *= exp(-abs(params.thickness / w.z));
                }

                if (z == 0.0) {
                    let u = vec2<f32>(
                        random01(ray.pixel_index, random_base + 2u, ray.depth),
                        random01(ray.pixel_index, random_base + 3u, ray.depth),
                    );
                    if (kind == MATERIAL_KIND_COATED_DIFFUSE) {
                        let radius = sqrt(u.x);
                        let phi = 2.0 * PI * u.y;
                        var wi = vec3<f32>(
                            radius * cos(phi), radius * sin(phi), sqrt(max(0.0, 1.0 - u.x)),
                        );
                        if ((-w).z < 0.0) { wi.z = -wi.z; }
                        let diffuse_pdf = abs(wi.z) / PI;
                        if (diffuse_pdf == 0.0) { break; }
                        path_f *= bottom.values[0] / PI;
                        path_pdf *= diffuse_pdf;
                        w = wi;
                    } else {
                        let bottom_sample = sample_conductor_interface(bottom, -w, u);
                        if (bottom_sample.valid == 0u || bottom_sample.pdf == 0.0
                            || bottom_sample.wi.z == 0.0) { break; }
                        path_f *= bottom_sample.f;
                        path_pdf *= bottom_sample.pdf;
                        w = bottom_sample.wi;
                    }
                    path_f *= abs(w.z);
                } else {
                    bs = sample_dielectric_interface(
                        top,
                        -w,
                        random01(ray.pixel_index, random_base + 1u, ray.depth),
                        vec2<f32>(
                            random01(ray.pixel_index, random_base + 2u, ray.depth),
                            random01(ray.pixel_index, random_base + 3u, ray.depth),
                        ),
                        true,
                        true,
                    );
                    if (bs.valid == 0u || bs.pdf == 0.0 || bs.wi.z == 0.0) { break; }
                    path_f *= bs.f;
                    path_pdf *= bs.pdf;
                    w = bs.wi;
                    if (bs.transmission != 0u) {
                        result_f = path_f;
                        result_pdf = path_pdf;
                        result_wi = w;
                        result_valid = true;
                        break;
                    }
                    path_f *= abs(w.z);
                }
            }
        }
        if (!result_valid || result_pdf <= 0.0 || max_spectrum(result_f) <= 0.0) { return; }
        if (flip_wi) { result_wi = -result_wi; }
        let direction = normalize(
            tangent * result_wi.x + bitangent * result_wi.y + normal * result_wi.z,
        );
        var next_throughput = ray.throughput * result_f * abs(result_wi.z) / result_pdf;
        if (ray.depth >= 1u) {
            let rr_beta = max_spectrum(next_throughput) / max(ray.inv_w_u, 1e-7);
            let q = max(0.0, 1.0 - rr_beta);
            if (samples.indirect.w < q) { return; }
            next_throughput /= max(1.0 - q, 1e-7);
        }
        let next_ray = RayWorkItem(
            vec4<f32>(offset_ray_origin(surface.position.xyz, surface.position_error.xyz,
                surface.geometric_normal.xyz, direction), 1.0),
            vec4<f32>(direction, 0.0), next_throughput,
            surface.position, surface.position_error, surface.geometric_normal,
            vec4<f32>(normal, 0.0), ray.pixel_index, ray.depth + 1u,
            ray.inv_w_u, ray.inv_w_u / max(result_pdf, 1e-7), result_pdf,
            0u, 0u, 0u,
        );
        let next_index = atomicAdd(&queue_counters.next.count, 1u);
        if (next_index >= queue_counters.next.capacity) {
            atomicStore(&queue_counters.next.overflow, 1u);
            return;
        }
        store_ray_samples(ray.pixel_index, generate_ray_samples(ray.pixel_index, ray.depth + 1u));
        store_next_ray(next_index, next_ray);
        return;
    }
}
