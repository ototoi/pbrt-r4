fn conductor_interface_alpha(item: AttributesEvalWorkItem) -> vec2<f32> {
    var roughness = max(vec2<f32>(item.values[2].x, item.values[3].x), vec2<f32>(0.0));
    if (item.values[4].x != 0.0) { roughness = sqrt(roughness); }
    return roughness;
}

fn sample_conductor_interface(
    item: AttributesEvalWorkItem, wo: vec3<f32>, u: vec2<f32>,
) -> DielectricInterfaceSample {
    let alpha = conductor_interface_alpha(item);
    if (max(alpha.x, alpha.y) < 1e-3) {
        let wi = vec3<f32>(-wo.x, -wo.y, wo.z);
        let value = conductor_fresnel(abs(wo.z), item.values[0], item.values[1]) / abs(wi.z);
        return DielectricInterfaceSample(value, wi, 1.0, 1.0, 1u, 0u, 1u);
    }
    let bounded_alpha = max(alpha, vec2<f32>(1e-4));
    let wm = sample_visible_tr_wm(wo, bounded_alpha, u);
    let wi = normalize(-wo + 2.0 * dot(wo, wm) * wm);
    if (wo.z * wi.z <= 0.0) { return invalid_dielectric_interface_sample(); }
    let pdf = tr_visible_wm_pdf(wo, wm, bounded_alpha)
        / max(4.0 * abs(dot(wo, wm)), 1e-7);
    let value = conductor_fresnel(abs(dot(wo, wm)), item.values[0], item.values[1])
        * tr_distribution_d(wm, bounded_alpha)
        * tr_distribution_g(wo, wi, bounded_alpha)
        / max(abs(4.0 * wo.z * wi.z), 1e-7);
    return DielectricInterfaceSample(value, wi, pdf, 1.0, 1u, 0u, 0u);
}

fn conductor_interface_f(
    item: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
) -> vec4<f32> {
    let alpha = conductor_interface_alpha(item);
    if (max(alpha.x, alpha.y) < 1e-3 || wo.z * wi.z <= 0.0) { return vec4<f32>(0.0); }
    var wm = normalize(wo + wi);
    if (wm.z < 0.0) { wm = -wm; }
    let bounded_alpha = max(alpha, vec2<f32>(1e-4));
    return conductor_fresnel(abs(dot(wo, wm)), item.values[0], item.values[1])
        * tr_distribution_d(wm, bounded_alpha) * tr_distribution_g(wo, wi, bounded_alpha)
        / max(abs(4.0 * wo.z * wi.z), 1e-7);
}

fn conductor_interface_pdf(
    item: AttributesEvalWorkItem, wo: vec3<f32>, wi: vec3<f32>,
) -> f32 {
    let alpha = conductor_interface_alpha(item);
    if (max(alpha.x, alpha.y) < 1e-3 || wo.z * wi.z <= 0.0) { return 0.0; }
    var wm = normalize(wo + wi);
    if (wm.z < 0.0) { wm = -wm; }
    return tr_visible_wm_pdf(wo, wm, max(alpha, vec2<f32>(1e-4)))
        / max(4.0 * abs(dot(wo, wm)), 1e-7);
}

fn load_conductor_eta(material_node: u32, lambda: vec4<f32>) -> vec4<f32> { return load_material_spectrum(material_node, 0u, lambda); }

fn load_conductor_k(material_node: u32, lambda: vec4<f32>) -> vec4<f32> { return load_material_spectrum(material_node, 1u, lambda); }

fn load_conductor_roughness(material_node: u32) -> f32 { return load_material_scalar(material_node, 2u); }

fn load_conductor_reflectance_roughness(material_node: u32) -> f32 { return load_material_scalar(material_node, 1u); }
