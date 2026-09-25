fn dielectric_interface_alpha(item: AttributesEvalWorkItem) -> vec2<f32> {
    var roughness = max(vec2<f32>(item.values[1].x, item.values[2].x), vec2<f32>(0.0));
    if (item.values[3].x != 0.0) {
        roughness = sqrt(roughness);
    }
    return roughness;
}

fn invalid_dielectric_interface_sample() -> DielectricInterfaceSample {
    return DielectricInterfaceSample(vec4<f32>(0.0), vec3<f32>(0.0), 0.0, 1.0, 0u, 0u, 0u);
}

fn sample_smooth_dielectric_interface(
    wo: vec3<f32>, eta: f32, uc: f32, allow_reflection: bool, allow_transmission: bool,
) -> DielectricInterfaceSample {
    let fresnel = dielectric_fresnel(wo.z, eta);
    let pr = select(0.0, fresnel, allow_reflection);
    let pt = select(0.0, 1.0 - fresnel, allow_transmission);
    if (pr + pt == 0.0) { return invalid_dielectric_interface_sample(); }
    if (uc < pr / (pr + pt)) {
        let wi = vec3<f32>(-wo.x, -wo.y, wo.z);
        return DielectricInterfaceSample(
            vec4<f32>(fresnel / abs(wi.z)), wi, pr / (pr + pt), 1.0, 1u, 0u, 1u,
        );
    }
    var result = refract_interface(wo, vec3<f32>(0.0, 0.0, 1.0), eta);
    if (result.valid == 0u) { return result; }
    var ft = (1.0 - fresnel) / abs(result.wi.z);
    // WavefrontPathIntegrator transports radiance.
    ft = ft / (result.etap * result.etap);
    result.f = vec4<f32>(ft);
    result.pdf = pt / (pr + pt);
    result.specular = 1u;
    return result;
}

fn sample_rough_dielectric_interface(
    wo: vec3<f32>, eta: f32, alpha_input: vec2<f32>, uc: f32, u: vec2<f32>,
    allow_reflection: bool, allow_transmission: bool,
) -> DielectricInterfaceSample {
    let alpha = max(alpha_input, vec2<f32>(1e-4));
    let wm = sample_visible_tr_wm(wo, alpha, u);
    let fresnel = dielectric_fresnel(dot(wo, wm), eta);
    let pr = select(0.0, fresnel, allow_reflection);
    let pt = select(0.0, 1.0 - fresnel, allow_transmission);
    if (pr + pt == 0.0) { return invalid_dielectric_interface_sample(); }
    let wm_pdf = tr_visible_wm_pdf(wo, wm, alpha);
    if (uc < pr / (pr + pt)) {
        let wi = normalize(-wo + 2.0 * dot(wo, wm) * wm);
        if (wo.z * wi.z <= 0.0) { return invalid_dielectric_interface_sample(); }
        let pdf = wm_pdf / max(4.0 * abs(dot(wo, wm)), 1e-7) * pr / (pr + pt);
        let value = tr_distribution_d(wm, alpha) * tr_distribution_g(wo, wi, alpha)
            * fresnel / max(abs(4.0 * wi.z * wo.z), 1e-7);
        return DielectricInterfaceSample(vec4<f32>(value), wi, pdf, 1.0, 1u, 0u, 0u);
    }
    var result = refract_interface(wo, wm, eta);
    if (result.valid == 0u || wo.z * result.wi.z >= 0.0 || result.wi.z == 0.0) { return invalid_dielectric_interface_sample(); }
    let denominator = dot(result.wi, wm) + dot(wo, wm) / result.etap;
    let denominator2 = denominator * denominator;
    if (denominator2 == 0.0) { return invalid_dielectric_interface_sample(); }
    let dwm_dwi = abs(dot(result.wi, wm)) / denominator2;
    result.pdf = wm_pdf * dwm_dwi * pt / (pr + pt);
    var ft = (1.0 - fresnel) * tr_distribution_d(wm, alpha)
        * tr_distribution_g(wo, result.wi, alpha)
        * abs(dot(result.wi, wm) * dot(wo, wm)
            / max(abs(result.wi.z * wo.z) * denominator2, 1e-7));
    ft = ft / (result.etap * result.etap);
    result.f = vec4<f32>(ft);
    result.specular = 0u;
    return result;
}

fn sample_dielectric_interface(
    item: AttributesEvalWorkItem, wo: vec3<f32>, uc: f32, u: vec2<f32>,
    allow_reflection: bool, allow_transmission: bool,
) -> DielectricInterfaceSample {
    let eta = select(item.values[0].x, 1.0, item.values[0].x == 0.0);
    let alpha = dielectric_interface_alpha(item);
    if (eta == 1.0 || max(alpha.x, alpha.y) < 1e-3) {
        return sample_smooth_dielectric_interface(
            wo, eta, uc, allow_reflection, allow_transmission,
        );
    }
    return sample_rough_dielectric_interface(
        wo, eta, alpha, uc, u, allow_reflection, allow_transmission,
    );
}

fn dielectric_interface_f(
    item: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
) -> vec4<f32> {
    let eta = select(item.values[0].x, 1.0, item.values[0].x == 0.0);
    let alpha = dielectric_interface_alpha(item);
    if (eta == 1.0 || max(alpha.x, alpha.y) < 1e-3 || wo.z == 0.0 || wi.z == 0.0) {
        return vec4<f32>(0.0);
    }
    let reflection = wo.z * wi.z > 0.0;
    var etap = 1.0;
    if (!reflection) { etap = select(1.0 / eta, eta, wo.z > 0.0); }
    var wm = wi * etap + wo;
    if (dot(wm, wm) == 0.0) { return vec4<f32>(0.0); }
    wm = normalize(wm);
    if (wm.z < 0.0) { wm = -wm; }
    if (dot(wm, wi) * wi.z < 0.0 || dot(wm, wo) * wo.z < 0.0) {
        return vec4<f32>(0.0);
    }
    let bounded_alpha = max(alpha, vec2<f32>(1e-4));
    let fresnel = dielectric_fresnel(dot(wo, wm), eta);
    if (reflection) {
        let value = tr_distribution_d(wm, bounded_alpha)
            * tr_distribution_g(wo, wi, bounded_alpha) * fresnel
            / max(abs(4.0 * wi.z * wo.z), 1e-7);
        return vec4<f32>(value);
    }
    let denominator = dot(wi, wm) + dot(wo, wm) / etap;
    var value = tr_distribution_d(wm, bounded_alpha) * (1.0 - fresnel)
        * tr_distribution_g(wo, wi, bounded_alpha)
        * abs(dot(wi, wm) * dot(wo, wm)
            / max(abs(wi.z * wo.z) * denominator * denominator, 1e-7));
    value /= etap * etap;
    return vec4<f32>(value);
}

fn dielectric_interface_pdf(
    item: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
    allow_reflection: bool, allow_transmission: bool,
) -> f32 {
    let eta = select(item.values[0].x, 1.0, item.values[0].x == 0.0);
    let alpha = dielectric_interface_alpha(item);
    if (eta == 1.0 || max(alpha.x, alpha.y) < 1e-3 || wo.z == 0.0 || wi.z == 0.0) {
        return 0.0;
    }
    let reflection = wo.z * wi.z > 0.0;
    var etap = 1.0;
    if (!reflection) { etap = select(1.0 / eta, eta, wo.z > 0.0); }
    var wm = wi * etap + wo;
    if (dot(wm, wm) == 0.0) { return 0.0; }
    wm = normalize(wm);
    if (wm.z < 0.0) { wm = -wm; }
    let fresnel = dielectric_fresnel(dot(wo, wm), eta);
    let pr = select(0.0, fresnel, allow_reflection);
    let pt = select(0.0, 1.0 - fresnel, allow_transmission);
    if (pr + pt == 0.0) { return 0.0; }
    let wm_pdf = tr_visible_wm_pdf(wo, wm, max(alpha, vec2<f32>(1e-4)));
    if (reflection) {
        return wm_pdf / max(4.0 * abs(dot(wo, wm)), 1e-7) * pr / (pr + pt);
    }
    let denominator = dot(wi, wm) + dot(wo, wm) / etap;
    return wm_pdf * abs(dot(wi, wm)) / max(denominator * denominator, 1e-7)
        * pt / (pr + pt);
}

fn sample_dielectric_interface_importance(
    item: AttributesEvalWorkItem, wo: vec3<f32>, uc: f32, u: vec2<f32>,
    allow_reflection: bool, allow_transmission: bool,
) -> DielectricInterfaceSample {
    var sample = sample_dielectric_interface(
        item, wo, uc, u, allow_reflection, allow_transmission,
    );
    if (sample.valid != 0u && sample.transmission != 0u) {
        sample.f *= sample.etap * sample.etap;
    }
    return sample;
}

fn load_dielectric_eta(material_node: u32, lambda: vec4<f32>) -> vec4<f32> { return load_material_spectrum(material_node, 0u, lambda); }

fn dielectric_eta_is_constant(material_node: u32) -> bool { return spectrum_is_constant(load_material_attribute(material_node, 0u).index); }
