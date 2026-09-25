fn refract_interface(wo: vec3<f32>, normal_input: vec3<f32>, eta_input: f32) -> DielectricInterfaceSample {
    var cosine_i = dot(normal_input, wo);
    var eta = eta_input;
    var normal = normal_input;
    if (cosine_i < 0.0) {
        eta = 1.0 / eta;
        cosine_i = -cosine_i;
        normal = -normal;
    }
    let sin2_t = max(0.0, 1.0 - cosine_i * cosine_i) / (eta * eta);
    if (sin2_t >= 1.0) { return invalid_dielectric_interface_sample(); }
    let cosine_t = sqrt(max(0.0, 1.0 - sin2_t));
    var result = invalid_dielectric_interface_sample();
    result.wi = normalize(-wo / eta + (cosine_i / eta - cosine_t) * normal);
    result.etap = eta;
    result.valid = 1u;
    result.transmission = 1u;
    return result;
}

fn tr_distribution_d(wm: vec3<f32>, alpha: vec2<f32>) -> f32 {
    let cos2 = wm.z * wm.z;
    if (cos2 < 1e-16) { return 0.0; }
    let e = (wm.x * wm.x / (alpha.x * alpha.x)
        + wm.y * wm.y / (alpha.y * alpha.y)) / cos2;
    return 1.0 / (PI * alpha.x * alpha.y * cos2 * cos2 * (1.0 + e) * (1.0 + e));
}

fn tr_distribution_lambda(w: vec3<f32>, alpha: vec2<f32>) -> f32 {
    let wz2 = w.z * w.z;
    if (wz2 == 0.0) { return 0.0; }
    let alpha2_tan2 = (alpha.x * w.x) * (alpha.x * w.x)
        + (alpha.y * w.y) * (alpha.y * w.y);
    return 0.5 * (sqrt(1.0 + alpha2_tan2 / wz2) - 1.0);
}

fn tr_distribution_g1(w: vec3<f32>, alpha: vec2<f32>) -> f32 {
    return 1.0 / (1.0 + tr_distribution_lambda(w, alpha));
}

fn tr_distribution_g(wo: vec3<f32>, wi: vec3<f32>, alpha: vec2<f32>) -> f32 {
    return 1.0 / (1.0 + tr_distribution_lambda(wo, alpha) + tr_distribution_lambda(wi, alpha));
}

fn sample_visible_tr_wm(wo_input: vec3<f32>, alpha: vec2<f32>, u: vec2<f32>) -> vec3<f32> {
    var wh = normalize(vec3<f32>(alpha.x * wo_input.x, alpha.y * wo_input.y, wo_input.z));
    if (wh.z < 0.0) { wh = -wh; }
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if (wh.z < 0.99999) { t1 = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), wh)); }
    let t2 = cross(wh, t1);
    let radius = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    var p = vec2<f32>(radius * cos(phi), radius * sin(phi));
    let h = sqrt(max(0.0, 1.0 - p.x * p.x));
    p.y = mix(h, p.y, (1.0 + wh.z) * 0.5);
    let pz = sqrt(max(0.0, 1.0 - dot(p, p)));
    let nh = p.x * t1 + p.y * t2 + pz * wh;
    return normalize(vec3<f32>(alpha.x * nh.x, alpha.y * nh.y, max(1e-6, nh.z)));
}

fn tr_visible_wm_pdf(wo: vec3<f32>, wm: vec3<f32>, alpha: vec2<f32>) -> f32 {
    if (abs(wo.z) == 0.0) { return 0.0; }
    return tr_distribution_d(wm, alpha) * tr_distribution_g1(wo, alpha)
        * abs(dot(wo, wm)) / abs(wo.z);
}

fn conductor_fresnel(cosine_input: f32, eta: vec4<f32>, k: vec4<f32>) -> vec4<f32> {
    let c = clamp(abs(cosine_input), 0.0, 1.0);
    let c2 = c * c;
    let s2 = 1.0 - c2;
    let eta2 = eta * eta;
    let k2 = k * k;
    let t0 = eta2 - k2 - vec4<f32>(s2);
    let a2b2 = sqrt(t0 * t0 + 4.0 * eta2 * k2);
    let a = sqrt(max(vec4<f32>(0.0), 0.5 * (a2b2 + t0)));
    let t1 = a2b2 + vec4<f32>(c2);
    let t2 = 2.0 * c * a;
    let rs = (t1 - t2) / max(t1 + t2, vec4<f32>(1e-7));
    let t3 = c2 * a2b2 + vec4<f32>(s2 * s2);
    let t4 = t2 * s2;
    let rp = rs * (t3 - t4) / max(t3 + t4, vec4<f32>(1e-7));
    return 0.5 * (rs + rp);
}

fn dielectric_fresnel(cosine: f32, eta: f32) -> f32 {
    let c = clamp(abs(cosine), 0.0, 1.0);
    let e = select(eta, 1.0 / eta, cosine < 0.0);
    let sin2_t = max(0.0, 1.0 - c * c) / (e * e);
    if (sin2_t >= 1.0) { return 1.0; }
    let ct = sqrt(1.0 - sin2_t);
    let rp = (e * c - ct) / (e * c + ct);
    let rs = (c - e * ct) / (c + e * ct);
    return (rp * rp + rs * rs) * 0.5;
}

// pbrt-v4 HGPhaseFunction helpers used by the layered medium random walk.
fn hg_phase(cosine: f32, g: f32) -> f32 {
    let bounded_g = clamp(g, -0.99, 0.99);
    let gg = bounded_g * bounded_g;
    let denominator = max(1.0 + gg + 2.0 * bounded_g * cosine, 1e-7);
    return (1.0 - gg) / (4.0 * PI * denominator * sqrt(denominator));
}

fn sample_hg_cosine(u: f32, g: f32) -> f32 {
    let bounded_g = clamp(g, -0.99, 0.99);
    if (abs(bounded_g) < 1e-3) {
        return 1.0 - 2.0 * u;
    }
    let t = (1.0 - bounded_g * bounded_g)
        / max(1.0 + bounded_g - 2.0 * bounded_g * u, 1e-7);
    return clamp(
        -(1.0 + bounded_g * bounded_g - t * t) / (2.0 * bounded_g),
        -1.0,
        1.0,
    );
}

fn sample_hg_direction(reference: vec3<f32>, u: vec2<f32>, g: f32) -> vec3<f32> {
    let cosine = sample_hg_cosine(u.x, g);
    let sine = sqrt(max(0.0, 1.0 - cosine * cosine));
    let phi = 2.0 * PI * u.y;
    let tangent = make_tangent(reference);
    let bitangent = cross(reference, tangent);
    return normalize(
        tangent * (sine * cos(phi))
        + bitangent * (sine * sin(phi))
        + reference * cosine,
    );
}
