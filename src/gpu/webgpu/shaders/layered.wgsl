// pbrt-v4 bxdfs.h: LayeredBxDF<DielectricBxDF, DiffuseBxDF> with a smooth top.
// Pure local-space routines: no scene buffers or wavefront queue operations.
const LAYERED_PI: f32 = 3.141592653589793;
const SCATTER_REFLECTION: u32 = 1u;
const SCATTER_TRANSMISSION: u32 = 2u;
const SCATTER_DIFFUSE: u32 = 4u;
const SCATTER_GLOSSY: u32 = 8u;
const SCATTER_SPECULAR: u32 = 16u;
const LAYERED_F_STREAM: u32 = 0u;
const LAYERED_SAMPLE_STREAM: u32 = 1u;

struct LayeredSample {
    f: vec4<f32>,
    wi: vec4<f32>,
    pdf: f32,
    eta: f32,
    flags: u32,
    valid: u32,
};

fn layered_hash(value: u32) -> u32 {
    var h = value;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

// Each tuple component is hashed separately; local depth cannot overlap a
// neighbouring pixel, global bounce, estimator sample, or event dimension.
fn layered_random(seed: u32, stream: u32, sample_index: u32, depth: u32, event: u32) -> f32 {
    let a = layered_hash(seed ^ layered_hash(stream));
    let b = layered_hash(a ^ sample_index);
    let c = layered_hash(b ^ depth);
    return f32(layered_hash(c ^ event) >> 8u) / 16777216.0;
}

fn layered_max(v: vec4<f32>) -> f32 { return max_spectrum(v); }

fn layered_tr(dz: f32, w: vec3<f32>) -> f32 {
    if (abs(dz) <= 1.17549435e-38) { return 1.0; }
    return exp(-abs(dz / w.z));
}

fn layered_invalid() -> LayeredSample {
    return LayeredSample(vec4<f32>(0.0), vec4<f32>(0.0), 0.0, 1.0, 0u, 0u);
}

fn layered_fresnel(cosine: f32, eta: f32) -> f32 {
    let c = clamp(abs(cosine), 0.0, 1.0);
    let e = select(eta, 1.0 / eta, cosine < 0.0);
    let sin2_t = max(0.0, 1.0 - c * c) / (e * e);
    if (sin2_t >= 1.0) { return 1.0; }
    let ct = sqrt(1.0 - sin2_t);
    let rp = (e * c - ct) / (e * c + ct);
    let rs = (c - e * ct) / (c + e * ct);
    return (rp * rp + rs * rs) * 0.5;
}

// radiance=false is v4's opposite transport mode for the virtual light.
fn layered_top_sample(wo: vec3<f32>, eta: f32, uc: f32, mask: u32, radiance: bool) -> LayeredSample {
    if (wo.z == 0.0) { return layered_invalid(); }
    let r = layered_fresnel(wo.z, eta);
    let t = 1.0 - r;
    let pr = select(0.0, r, (mask & SCATTER_REFLECTION) != 0u);
    let pt = select(0.0, t, (mask & SCATTER_TRANSMISSION) != 0u);
    if (pr + pt == 0.0) { return layered_invalid(); }
    if (uc < pr / (pr + pt)) {
        return LayeredSample(vec4<f32>(vec3<f32>(r / abs(wo.z)), 0.0),
            vec4<f32>(-wo.xy, wo.z, 0.0), pr / (pr + pt), 1.0,
            SCATTER_REFLECTION | SCATTER_SPECULAR, 1u);
    }
    let etap = select(eta, 1.0 / eta, wo.z < 0.0);
    let sin2_t = max(0.0, 1.0 - wo.z * wo.z) / (etap * etap);
    if (sin2_t >= 1.0) { return layered_invalid(); }
    let wi = vec3<f32>(-wo.xy / etap, -sign(wo.z) * sqrt(1.0 - sin2_t));
    var ft = t / abs(wi.z);
    if (radiance) { ft /= etap * etap; }
    return LayeredSample(vec4<f32>(vec3<f32>(ft), 0.0), vec4<f32>(wi, 0.0),
        pt / (pr + pt), etap, SCATTER_TRANSMISSION | SCATTER_SPECULAR, 1u);
}

fn layered_bottom_sample(wo: vec3<f32>, reflectance: vec4<f32>, u: vec2<f32>) -> LayeredSample {
    let r = sqrt(u.x);
    let phi = 2.0 * LAYERED_PI * u.y;
    let wi = vec3<f32>(r * cos(phi), r * sin(phi),
        select(1.0, -1.0, wo.z < 0.0) * sqrt(max(0.0, 1.0 - u.x)));
    if (wi.z == 0.0 || layered_max(reflectance) == 0.0) { return layered_invalid(); }
    return LayeredSample(reflectance / LAYERED_PI, vec4<f32>(wi, 0.0),
        abs(wi.z) / LAYERED_PI, 1.0, SCATTER_REFLECTION | SCATTER_DIFFUSE, 1u);
}

fn layered_hg(cosine: f32, g_input: f32) -> f32 {
    // Same stability clamp as v4 util/scattering.h::HenyeyGreenstein.
    let g = clamp(g_input, -0.99, 0.99);
    let d = 1.0 + g * g + 2.0 * g * cosine;
    return (1.0 - g * g) / (4.0 * LAYERED_PI * d * sqrt(max(0.0, d)));
}

fn layered_hg_sample(wo: vec3<f32>, g_input: f32, u: vec2<f32>) -> vec3<f32> {
    let g = clamp(g_input, -0.99, 0.99);
    var c = 1.0 - 2.0 * u.x;
    if (abs(g) >= 1e-3) {
        let a = (1.0 - g * g) / (1.0 + g - 2.0 * g * u.x);
        c = -(1.0 + g * g - a * a) / (2.0 * g);
    }
    let s = sqrt(max(0.0, 1.0 - c * c));
    let phi = 2.0 * LAYERED_PI * u.y;
    // v4 Frame::FromZ / CoordinateSystem.
    let sign_z = select(-1.0, 1.0, wo.z >= 0.0);
    let a = -1.0 / (sign_z + wo.z);
    let b = wo.x * wo.y * a;
    let x = vec3<f32>(1.0 + sign_z * wo.x * wo.x * a, sign_z * b, -sign_z * wo.x);
    let y = vec3<f32>(b, sign_z + wo.y * wo.y * a, -wo.y);
    return x * (s * cos(phi)) + y * (s * sin(phi)) + wo * c;
}

fn layered_sample(data: LayeredParams, eta: f32, reflectance: vec4<f32>,
    wo_input: vec3<f32>, uc: f32, u: vec2<f32>, seed: u32) -> LayeredSample {
    let flip = data.two_sided != 0u && wo_input.z < 0.0;
    let wo = select(wo_input, -wo_input, flip);
    if (wo.z == 0.0) { return layered_invalid(); }
    if (wo.z < 0.0) { return layered_bottom_sample(wo, reflectance, u); }
    var bs = layered_top_sample(wo, eta, uc, SCATTER_REFLECTION | SCATTER_TRANSMISSION, true);
    if (bs.valid == 0u) { return bs; }
    if ((bs.flags & SCATTER_REFLECTION) != 0u) {
        if (flip) { bs.wi = -bs.wi; }
        return bs;
    }
    var f = bs.f * abs(bs.wi.z);
    var pdf = bs.pdf;
    var w = bs.wi.xyz;
    var z = data.thickness;
    var specular_path = true;
    for (var depth = 0u; depth < data.max_depth; depth++) {
        let rr_beta = layered_max(f) / pdf;
        if (depth > 3u && rr_beta < 0.25) {
            let q = max(0.0, 1.0 - rr_beta);
            if (layered_random(seed, LAYERED_SAMPLE_STREAM, 0u, depth, 0u) < q) { return layered_invalid(); }
            pdf *= 1.0 - q;
        }
        if (w.z == 0.0) { return layered_invalid(); }
        if (layered_max(data.albedo) > 0.0) {
            let dz = -log(1.0 - layered_random(seed, LAYERED_SAMPLE_STREAM, 0u, depth, 1u)) * abs(w.z);
            let zp = z + sign(w.z) * dz;
            if (zp == z) { return layered_invalid(); }
            if (zp > 0.0 && zp < data.thickness) {
                let phase_u = vec2<f32>(layered_random(seed, LAYERED_SAMPLE_STREAM, 0u, depth, 2u),
                    layered_random(seed, LAYERED_SAMPLE_STREAM, 0u, depth, 3u));
                let wi = layered_hg_sample(-w, data.g, phase_u);
                let p = layered_hg(dot(-w, wi), data.g);
                if (p == 0.0 || wi.z == 0.0) { return layered_invalid(); }
                f *= data.albedo * p;
                pdf *= p;
                specular_path = false;
                w = wi;
                z = zp;
                continue;
            }
            z = clamp(zp, 0.0, data.thickness);
        } else {
            z = select(0.0, data.thickness, z == 0.0);
            f *= layered_tr(data.thickness, w);
        }
        let interface_uc = layered_random(seed, LAYERED_SAMPLE_STREAM, 0u, depth, 4u);
        let interface_u = vec2<f32>(layered_random(seed, LAYERED_SAMPLE_STREAM, 0u, depth, 5u),
            layered_random(seed, LAYERED_SAMPLE_STREAM, 0u, depth, 6u));
        if (z == 0.0) { bs = layered_bottom_sample(-w, reflectance, interface_u); }
        else { bs = layered_top_sample(-w, eta, interface_uc, SCATTER_REFLECTION | SCATTER_TRANSMISSION, true); }
        if (bs.valid == 0u || bs.wi.z == 0.0) { return layered_invalid(); }
        f *= bs.f;
        pdf *= bs.pdf;
        specular_path = specular_path && (bs.flags & SCATTER_SPECULAR) != 0u;
        w = bs.wi.xyz;
        if ((bs.flags & SCATTER_TRANSMISSION) != 0u) {
            let flags = SCATTER_REFLECTION | select(SCATTER_GLOSSY, SCATTER_SPECULAR, specular_path);
            return LayeredSample(f, vec4<f32>(select(w, -w, flip), 0.0), pdf, 1.0, flags, 1u);
        }
        f *= abs(w.z);
    }
    return layered_invalid();
}

fn layered_f(data: LayeredParams, eta: f32, reflectance: vec4<f32>,
    wo_input: vec3<f32>, wi_input: vec3<f32>, seed: u32) -> vec4<f32> {
    let flip = data.two_sided != 0u && wo_input.z < 0.0;
    let wo = select(wo_input, -wo_input, flip);
    let wi = select(wi_input, -wi_input, flip);
    if (wo.z * wi.z <= 0.0) { return vec4<f32>(0.0); }
    if (wo.z < 0.0) { return reflectance / LAYERED_PI; }
    let wos = layered_top_sample(wo, eta, 0.0, SCATTER_TRANSMISSION, true);
    let wis = layered_top_sample(wi, eta, 0.0, SCATTER_TRANSMISSION, false);
    if (wos.valid == 0u || wis.valid == 0u) { return vec4<f32>(0.0); }
    // The smooth exit has no non-delta f/PDF. Thus v4's two PowerHeuristic
    // branches reduce to weight=1, and the complementary exit-f terms vanish.
    var result = vec4<f32>(0.0);
    for (var sample_index = 0u; sample_index < data.n_samples; sample_index++) {
        var beta = wos.f * abs(wos.wi.z) / wos.pdf;
        var z = data.thickness;
        var w = wos.wi.xyz;
        for (var depth = 0u; depth < data.max_depth; depth++) {
            if (depth > 3u && layered_max(beta) < 0.25) {
                let q = max(0.0, 1.0 - layered_max(beta));
                if (layered_random(seed, LAYERED_F_STREAM, sample_index, depth, 0u) < q) { break; }
                beta /= 1.0 - q;
            }
            if (layered_max(data.albedo) == 0.0) {
                z = select(0.0, data.thickness, z == 0.0);
                beta *= layered_tr(data.thickness, w);
            } else {
                let dz = -log(1.0 - layered_random(seed, LAYERED_F_STREAM, sample_index, depth, 1u)) * abs(w.z);
                let zp = z + sign(w.z) * dz;
                if (zp == z) { continue; }
                if (zp > 0.0 && zp < data.thickness) {
                    result += beta * data.albedo * layered_hg(dot(-w, -wis.wi.xyz), data.g)
                        * layered_tr(zp - data.thickness, wis.wi.xyz) * wis.f / wis.pdf;
                    let phase_u = vec2<f32>(layered_random(seed, LAYERED_F_STREAM, sample_index, depth, 2u),
                        layered_random(seed, LAYERED_F_STREAM, sample_index, depth, 3u));
                    let phase_wi = layered_hg_sample(-w, data.g, phase_u);
                    if (phase_wi.z == 0.0 || layered_hg(dot(-w, phase_wi), data.g) == 0.0) { continue; }
                    beta *= data.albedo;
                    w = phase_wi;
                    z = zp;
                    continue;
                }
                z = clamp(zp, 0.0, data.thickness);
            }
            if (z == data.thickness) {
                let bs = layered_top_sample(-w, eta, 0.0, SCATTER_REFLECTION, true);
                if (bs.valid == 0u) { break; }
                beta *= bs.f * abs(bs.wi.z) / bs.pdf;
                w = bs.wi.xyz;
            } else {
                result += beta * (reflectance / LAYERED_PI) * abs(wis.wi.z)
                    * layered_tr(data.thickness, wis.wi.xyz) * wis.f / wis.pdf;
                let u = vec2<f32>(layered_random(seed, LAYERED_F_STREAM, sample_index, depth, 5u),
                    layered_random(seed, LAYERED_F_STREAM, sample_index, depth, 6u));
                let bs = layered_bottom_sample(-w, reflectance, u);
                if (bs.valid == 0u) { break; }
                beta *= bs.f * abs(bs.wi.z) / bs.pdf;
                w = bs.wi.xyz;
            }
        }
    }
    return result / f32(data.n_samples);
}

fn layered_pdf(data: LayeredParams, eta: f32, wo_input: vec3<f32>, wi_input: vec3<f32>) -> f32 {
    let flip = data.two_sided != 0u && wo_input.z < 0.0;
    let wo = select(wo_input, -wo_input, flip);
    let wi = select(wi_input, -wi_input, flip);
    var estimate = 0.0;
    if (wo.z * wi.z > 0.0) {
        if (wo.z < 0.0) { estimate = abs(wi.z) / LAYERED_PI; }
        else {
            let wos = layered_top_sample(wo, eta, 0.0, SCATTER_TRANSMISSION, true);
            let wis = layered_top_sample(wi, eta, 0.0, SCATTER_TRANSMISSION, false);
            if (wos.valid != 0u && wis.valid != 0u) { estimate = abs(wis.wi.z) / LAYERED_PI; }
        }
    }
    // v4 LayeredBxDF::PDF uses this mixture even for the smooth TRT estimate.
    return 0.1 / (4.0 * LAYERED_PI) + 0.9 * estimate;
}
