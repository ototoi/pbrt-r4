fn load_layered_params(root: AttributesEvalWorkItem, kind: u32) -> LayeredParams {
    var params: LayeredParams;
    params.thickness = root.values[0].x;
    params.g = root.values[2].x;
    params.max_depth = root.values[3].x;
    params.n_samples = root.values[4].x;
    params.albedo = root.values[1];
    if (kind == MATERIAL_KIND_COATED_DIFFUSE) { params.albedo = root.values[5]; }
    return params;
}

fn layered_bottom_f(
    kind: u32, bottom: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
) -> vec4<f32> {
    if (kind == MATERIAL_KIND_COATED_DIFFUSE) {
        return select(vec4<f32>(0.0), bottom.values[0] / PI, wo.z * wi.z > 0.0);
    }
    return conductor_interface_f(bottom, wo, wi);
}

fn layered_bottom_pdf(
    kind: u32, bottom: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
) -> f32 {
    if (kind == MATERIAL_KIND_COATED_DIFFUSE) {
        return select(0.0, abs(wi.z) / PI, wo.z * wi.z > 0.0);
    }
    return conductor_interface_pdf(bottom, wo, wi);
}

fn sample_layered_bottom(
    kind: u32, bottom: AttributesEvalWorkItem, wo: vec3<f32>, u: vec2<f32>,
) -> DielectricInterfaceSample {
    if (kind == MATERIAL_KIND_COATED_CONDUCTOR) {
        return sample_conductor_interface(bottom, wo, u);
    }
    let radius = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    var wi = vec3<f32>(radius * cos(phi), radius * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
    if (wo.z < 0.0) { wi.z = -wi.z; }
    let pdf = abs(wi.z) / PI;
    if (pdf == 0.0) { return invalid_dielectric_interface_sample(); }
    return DielectricInterfaceSample(bottom.values[0] / PI, wi, pdf, 1.0, 1u, 0u, 0u);
}

fn power_heuristic_one(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    return a2 / max(a2 + pdf_b * pdf_b, 1e-30);
}

fn layered_tr(distance: f32, w: vec3<f32>) -> f32 {
    if (abs(distance) <= 1.17549435e-38) { return 1.0; }
    return exp(-abs(distance / w.z));
}

fn evaluate_layered_f(
    root: AttributesEvalWorkItem, kind: u32, wo_input: vec3<f32>, wi_input: vec3<f32>,
    pixel_index: u32, path_depth: u32,
) -> vec4<f32> {
    if (wo_input.z == 0.0 || wi_input.z == 0.0 || wo_input.z * wi_input.z <= 0.0) {
        return vec4<f32>(0.0);
    }
    var wo = wo_input;
    var wi = wi_input;
    if (wo.z < 0.0) { wo = -wo; wi = -wi; }
    let top = load_attributes_eval_work_item(root.child_work_item0);
    let bottom = load_attributes_eval_work_item(root.child_work_item1);
    let params = load_layered_params(root, kind);
    let n_samples = max(1u, u32(params.n_samples));
    let max_depth = max(1u, u32(params.max_depth));
    var result = f32(n_samples) * dielectric_interface_f(top, wo, wi);
    let top_rough = max(dielectric_interface_alpha(top).x, dielectric_interface_alpha(top).y) >= 1e-3;
    let bottom_rough = kind == MATERIAL_KIND_COATED_DIFFUSE
        || max(conductor_interface_alpha(bottom).x, conductor_interface_alpha(bottom).y) >= 1e-3;
    for (var sample_index = 0u; sample_index < n_samples; sample_index++) {
        let sample_base = 64u + sample_index * 256u;
        let wos = sample_dielectric_interface(
            top, wo,
            random01(pixel_index, sample_base, path_depth),
            vec2<f32>(random01(pixel_index, sample_base + 1u, path_depth),
                random01(pixel_index, sample_base + 2u, path_depth)),
            false, true,
        );
        if (wos.valid == 0u || wos.pdf == 0.0 || wos.wi.z == 0.0) { continue; }
        let wis = sample_dielectric_interface_importance(
            top, wi,
            random01(pixel_index, sample_base + 3u, path_depth),
            vec2<f32>(random01(pixel_index, sample_base + 4u, path_depth),
                random01(pixel_index, sample_base + 5u, path_depth)),
            false, true,
        );
        if (wis.valid == 0u || wis.pdf == 0.0 || wis.wi.z == 0.0) { continue; }
        var beta = wos.f * abs(wos.wi.z) / wos.pdf;
        var z = max(params.thickness, 1.17549435e-38);
        var w = wos.wi;
        for (var layer_depth = 0u; layer_depth < max_depth; layer_depth++) {
            let random_base = sample_base + 8u + layer_depth * 6u;
            if (layer_depth > 3u && max_spectrum(beta) < 0.25) {
                let q = max(0.0, 1.0 - max_spectrum(beta));
                if (random01(pixel_index, random_base, path_depth) < q) { break; }
                beta /= max(1.0 - q, 1e-7);
            }
            if (w.z == 0.0) { break; }
            if (max_spectrum(params.albedo) > 0.0) {
                let dz = sample_layered_exponential(
                    random01(pixel_index, random_base + 1u, path_depth), 1.0 / abs(w.z),
                );
                let zp = select(z - dz, z + dz, w.z > 0.0);
                if (zp == z) { continue; }
                if (zp > 0.0 && zp < params.thickness) {
                    let phase_to_wi = hg_phase(dot(normalize(-w), normalize(-wis.wi)), params.g);
                    var wt = 1.0;
                    if (top_rough) { wt = power_heuristic_one(wis.pdf, phase_to_wi); }
                    result += beta * params.albedo * phase_to_wi * wt
                        * layered_tr(zp - params.thickness, wis.wi) * wis.f / wis.pdf;
                    let phase_wi = sample_hg_direction(
                        normalize(-w),
                        vec2<f32>(random01(pixel_index, random_base + 2u, path_depth),
                            random01(pixel_index, random_base + 3u, path_depth)), params.g,
                    );
                    let phase_pdf = hg_phase(dot(normalize(-w), phase_wi), params.g);
                    if (phase_pdf == 0.0 || phase_wi.z == 0.0) { continue; }
                    beta *= params.albedo;
                    w = phase_wi;
                    z = zp;
                    if (w.z > 0.0 && top_rough) {
                        let f_exit = dielectric_interface_f(top, -w, wi);
                        let exit_pdf = dielectric_interface_pdf(top, -w, wi, false, true);
                        result += beta * layered_tr(zp - params.thickness, w) * f_exit
                            * power_heuristic_one(phase_pdf, exit_pdf);
                    }
                    continue;
                }
                z = clamp(zp, 0.0, params.thickness);
            } else {
                z = select(params.thickness, 0.0, z == params.thickness);
                beta *= layered_tr(params.thickness, w);
            }
            if (z == params.thickness) {
                let reflected = sample_dielectric_interface(
                    top, -w, random01(pixel_index, random_base + 1u, path_depth),
                    vec2<f32>(random01(pixel_index, random_base + 2u, path_depth),
                        random01(pixel_index, random_base + 3u, path_depth)), true, false,
                );
                if (reflected.valid == 0u || reflected.pdf == 0.0 || reflected.wi.z == 0.0) { break; }
                beta *= reflected.f * abs(reflected.wi.z) / reflected.pdf;
                w = reflected.wi;
            } else {
                if (bottom_rough) {
                    var wt = 1.0;
                    if (top_rough) {
                        wt = power_heuristic_one(wis.pdf,
                            layered_bottom_pdf(kind, bottom, -w, -wis.wi));
                    }
                    result += beta * layered_bottom_f(kind, bottom, -w, -wis.wi)
                        * abs(wis.wi.z) * wt * layered_tr(params.thickness, wis.wi)
                        * wis.f / wis.pdf;
                }
                let reflected = sample_layered_bottom(
                    kind, bottom, -w,
                    vec2<f32>(random01(pixel_index, random_base + 2u, path_depth),
                        random01(pixel_index, random_base + 3u, path_depth)),
                );
                if (reflected.valid == 0u || reflected.pdf == 0.0 || reflected.wi.z == 0.0) { break; }
                beta *= reflected.f * abs(reflected.wi.z) / reflected.pdf;
                w = reflected.wi;
                if (top_rough) {
                    let f_exit = dielectric_interface_f(top, -w, wi);
                    let exit_pdf = dielectric_interface_pdf(top, -w, wi, false, true);
                    result += beta * layered_tr(params.thickness, w) * f_exit
                        * power_heuristic_one(reflected.pdf, exit_pdf);
                }
            }
        }
    }
    return result / f32(n_samples);
}

fn evaluate_layered_pdf(
    root: AttributesEvalWorkItem, kind: u32, wo_input: vec3<f32>, wi_input: vec3<f32>,
    pixel_index: u32, path_depth: u32,
) -> f32 {
    if (wo_input.z == 0.0 || wi_input.z == 0.0 || wo_input.z * wi_input.z <= 0.0) {
        return 0.0;
    }
    var wo = wo_input;
    var wi = wi_input;
    if (wo.z < 0.0) { wo = -wo; wi = -wi; }
    let top = load_attributes_eval_work_item(root.child_work_item0);
    let bottom = load_attributes_eval_work_item(root.child_work_item1);
    let n_samples = max(1u, u32(load_layered_params(root, kind).n_samples));
    let top_rough = max(dielectric_interface_alpha(top).x, dielectric_interface_alpha(top).y) >= 1e-3;
    let bottom_rough = kind == MATERIAL_KIND_COATED_DIFFUSE
        || max(conductor_interface_alpha(bottom).x, conductor_interface_alpha(bottom).y) >= 1e-3;
    var pdf_sum = f32(n_samples) * dielectric_interface_pdf(top, wo, wi, true, false);
    for (var sample_index = 0u; sample_index < n_samples; sample_index++) {
        let random_base = 20000u + sample_index * 8u;
        let wos = sample_dielectric_interface(
            top, wo,
            random01(pixel_index, random_base, path_depth),
            vec2<f32>(random01(pixel_index, random_base + 1u, path_depth),
                random01(pixel_index, random_base + 2u, path_depth)),
            false, true,
        );
        if (wos.valid == 0u || wos.pdf == 0.0 || wos.wi.z == 0.0) { continue; }
        let wis = sample_dielectric_interface_importance(
            top, wi,
            random01(pixel_index, random_base + 3u, path_depth),
            vec2<f32>(random01(pixel_index, random_base + 4u, path_depth),
                random01(pixel_index, random_base + 5u, path_depth)),
            false, true,
        );
        if (wis.valid == 0u || wis.pdf == 0.0 || wis.wi.z == 0.0) { continue; }
        let reflection_pdf = layered_bottom_pdf(kind, bottom, -wos.wi, -wis.wi);
        if (!top_rough) {
            pdf_sum += reflection_pdf;
            continue;
        }
        let reflected = sample_layered_bottom(
            kind, bottom, -wos.wi,
            vec2<f32>(random01(pixel_index, random_base + 6u, path_depth),
                random01(pixel_index, random_base + 7u, path_depth)),
        );
        if (reflected.valid == 0u || reflected.pdf == 0.0 || reflected.wi.z == 0.0) {
            continue;
        }
        let transmission_pdf = dielectric_interface_pdf(
            top, -reflected.wi, wi, false, true,
        );
        if (!bottom_rough) {
            pdf_sum += transmission_pdf;
        } else {
            pdf_sum += power_heuristic_one(wis.pdf, reflection_pdf) * reflection_pdf;
            pdf_sum += power_heuristic_one(reflected.pdf, transmission_pdf) * transmission_pdf;
        }
    }
    return mix(1.0 / (4.0 * PI), pdf_sum / f32(n_samples), 0.9);
}

fn sample_layered_exponential(u: f32, rate: f32) -> f32 {
    return -log(max(1.0 - min(u, 0.99999994), 1e-7)) / max(rate, 1e-7);
}
