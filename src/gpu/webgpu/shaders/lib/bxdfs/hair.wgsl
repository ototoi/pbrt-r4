// pbrt-v4 HairBxDF (bxdfs.h:921, bxdfs.cpp:275) translated via the Rust
// CPU port pbrt-r4/src/bxdfs/hair.rs, using vec4<f32> per-wavelength lanes
// like the conductor BxDF. No numerical adjustments of its own.
//
// The attributes layout follows docs/webgpu-hair-material-design_ja.md §2.2:
// values[0].x beta_m, [1].x beta_n, [2].x alpha, [3].x eta, [4] resolved
// sigma_a, [8].x sigma_a mode (0=sigma_a, 1=reflectance, 2=pigment).

const HAIR_P_MAX: u32 = 3u;
const HAIR_SQRT_PI_OVER8: f32 = 0.626657069;

struct HairBxdfState {
    h: f32,
    gamma_o: f32,
    eta: f32,
    sigma_a: vec4<f32>,
    v: array<f32, 4>,
    s: f32,
    sin2k_alpha: array<f32, 3>,
    cos2k_alpha: array<f32, 3>,
}

struct HairBsdfSample {
    f: vec4<f32>,
    wi: vec3<f32>,
    pdf: f32,
    valid: u32,
}

// pbrt-v4 HairMaterial::GetBxDF: h = -1 + 2 * v on the diced curve strips.
fn hair_bxdf_make(item: AttributesEvalWorkItem, h: f32) -> HairBxdfState {
    let beta_m = item.values[0].x;
    let beta_n = item.values[1].x;
    let alpha = item.values[2].x;
    let eta = item.values[3].x;

    let gamma_o = hair_safe_asin(h);

    // HairBxDF::new: v[0] from beta_m's polynomial, then 1/4 and 4x.
    let v0 = sqr_f32(0.726 * beta_m + 0.812 * sqr_f32(beta_m) + 3.7 * pow(beta_m, 20.0));
    var v = array<f32, 4>(v0, 0.25 * v0, 4.0 * v0, 4.0 * v0);

    let s = HAIR_SQRT_PI_OVER8
        * (0.265 * beta_n + 1.194 * sqr_f32(beta_n) + 5.372 * pow(beta_n, 22.0));

    let sin0 = sin(radians_f32(alpha));
    let cos0 = sqrt(max(0.0, 1.0 - sqr_f32(sin0)));
    var sin2k_alpha = array<f32, 3>(sin0, 0.0, 0.0);
    var cos2k_alpha = array<f32, 3>(cos0, 0.0, 0.0);
    for (var i = 1u; i < 3u; i = i + 1u) {
        sin2k_alpha[i] = 2.0 * cos2k_alpha[i - 1u] * sin2k_alpha[i - 1u];
        cos2k_alpha[i] = sqr_f32(cos2k_alpha[i - 1u]) - sqr_f32(sin2k_alpha[i - 1u]);
    }

    return HairBxdfState(h, gamma_o, eta, item.values[4], v, s, sin2k_alpha, cos2k_alpha);
}

// HairBxDF::f.
fn hair_bxdf_f(b: HairBxdfState, wo: vec3<f32>, wi: vec3<f32>) -> vec4<f32> {
    let sin_theta_o = wo.x;
    let cos_theta_o = sqrt(max(0.0, 1.0 - sqr_f32(sin_theta_o)));
    let phi_o = atan2(wo.z, wo.y);

    let sin_theta_i = wi.x;
    let cos_theta_i = sqrt(max(0.0, 1.0 - sqr_f32(sin_theta_i)));
    let phi_i = atan2(wi.z, wi.y);

    let sin_theta_t = sin_theta_o / b.eta;
    let cos_theta_t = sqrt(max(0.0, 1.0 - sqr_f32(sin_theta_t)));

    let etap = sqrt(max(0.0, b.eta * b.eta - sqr_f32(sin_theta_o))) / cos_theta_o;
    let sin_gamma_t = b.h / etap;
    let cos_gamma_t = sqrt(max(0.0, 1.0 - sqr_f32(sin_gamma_t)));
    let gamma_t = hair_safe_asin(sin_gamma_t);

    let t = exp(-b.sigma_a * (2.0 * cos_gamma_t / cos_theta_t));
    let ap = hair_ap(cos_theta_o, b.eta, b.h, t);

    let phi = phi_i - phi_o;
    var fsum = vec4<f32>(0.0);
    for (var p = 0u; p < HAIR_P_MAX; p = p + 1u) {
        let theta_op = hair_compute_theta_op(b, p, sin_theta_o, cos_theta_o);
        let cos_theta_op = abs(theta_op.y);
        fsum += ap[p]
            * vec4<f32>(hair_mp(cos_theta_i, cos_theta_op, sin_theta_i, theta_op.x, b.v[p]))
            * vec4<f32>(hair_np(phi, p, b.s, b.gamma_o, gamma_t));
    }
    fsum += ap[HAIR_P_MAX]
        * vec4<f32>(hair_mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, b.v[HAIR_P_MAX]) / (2.0 * PI));

    let abs_cos_theta_wi = abs(wi.z);
    if (abs_cos_theta_wi > 0.0) {
        fsum /= abs_cos_theta_wi;
    }
    return fsum;
}

// HairBxDF::Sample_f. u comes from the ray's indirect samples.
fn hair_bxdf_sample_f(b: HairBxdfState, wo: vec3<f32>, u: vec2<f32>) -> HairBsdfSample {
    let invalid = HairBsdfSample(vec4<f32>(0.0), vec3<f32>(0.0), 0.0, 0u);

    let sin_theta_o = wo.x;
    let cos_theta_o = sqrt(max(0.0, 1.0 - sqr_f32(sin_theta_o)));
    let phi_o = atan2(wo.z, wo.y);

    let uu0 = hair_demux_float(u.x);
    let uu1 = hair_demux_float(u.y);
    var uu0x = uu0.x;
    let ap_pdf = hair_compute_ap_pdf(b, cos_theta_o);

    var p = HAIR_P_MAX;
    for (var i = 0u; i <= HAIR_P_MAX; i = i + 1u) {
        if (uu0x < ap_pdf[i]) {
            p = i;
            break;
        }
        uu0x -= ap_pdf[i];
    }

    let theta_op = hair_compute_theta_op(b, p, sin_theta_o, cos_theta_o);
    let sin_theta_op = theta_op.x;
    let cos_theta_op = theta_op.y;
    let uu1x = max(uu1.x, 1e-5);
    let cos_theta = 1.0 + b.v[p] * log(uu1x + (1.0 - uu1x) * exp(-2.0 / b.v[p]));
    let sin_theta = sqrt(max(0.0, 1.0 - sqr_f32(cos_theta)));
    let cos_phi = cos(2.0 * PI * uu1.y);
    let sin_theta_i = -cos_theta * sin_theta_op + sin_theta * cos_phi * cos_theta_op;
    let cos_theta_i = sqrt(max(0.0, 1.0 - sqr_f32(sin_theta_i)));

    let etap = sqrt(max(0.0, b.eta * b.eta - sqr_f32(sin_theta_o))) / cos_theta_o;
    let sin_gamma_t = b.h / etap;
    let gamma_t = hair_safe_asin(sin_gamma_t);
    let dphi = select(
        2.0 * PI * uu0.y,
        hair_phi(p, b.gamma_o, gamma_t) + hair_sample_trimmed_logistic(uu0.y, b.s, -PI, PI),
        p < HAIR_P_MAX,
    );

    let phi_i = phi_o + dphi;
    let wi = vec3<f32>(
        sin_theta_i,
        cos_theta_i * cos(phi_i),
        cos_theta_i * sin(phi_i),
    );

    let pdf = hair_bxdf_pdf(b, wo, wi);
    if (pdf <= 0.0) {
        return invalid;
    }
    return HairBsdfSample(hair_bxdf_f(b, wo, wi), wi, pdf, 1u);
}

// HairBxDF::Pdf.
fn hair_bxdf_pdf(b: HairBxdfState, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
    let sin_theta_o = wo.x;
    let cos_theta_o = sqrt(max(0.0, 1.0 - sqr_f32(sin_theta_o)));
    let phi_o = atan2(wo.z, wo.y);

    let sin_theta_i = wi.x;
    let cos_theta_i = sqrt(max(0.0, 1.0 - sqr_f32(sin_theta_i)));
    let phi_i = atan2(wi.z, wi.y);

    let etap = sqrt(max(0.0, b.eta * b.eta - sqr_f32(sin_theta_o))) / cos_theta_o;
    let sin_gamma_t = b.h / etap;
    let gamma_t = hair_safe_asin(sin_gamma_t);

    let ap_pdf = hair_compute_ap_pdf(b, cos_theta_o);
    let phi = phi_i - phi_o;
    var pdf = 0.0;
    for (var p = 0u; p < HAIR_P_MAX; p = p + 1u) {
        let theta_op = hair_compute_theta_op(b, p, sin_theta_o, cos_theta_o);
        let cos_theta_op = abs(theta_op.y);
        pdf += hair_mp(cos_theta_i, cos_theta_op, sin_theta_i, theta_op.x, b.v[p])
            * ap_pdf[p] * hair_np(phi, p, b.s, b.gamma_o, gamma_t);
    }
    pdf += hair_mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, b.v[HAIR_P_MAX])
        * ap_pdf[HAIR_P_MAX] / (2.0 * PI);
    return pdf;
}

// HairBxDF::ComputeAPPDF.
fn hair_compute_ap_pdf(b: HairBxdfState, cos_theta_o: f32) -> array<f32, 4> {
    let sin_theta_o = sqrt(max(0.0, 1.0 - cos_theta_o * cos_theta_o));
    let sin_theta_t = sin_theta_o / b.eta;
    let cos_theta_t = sqrt(max(0.0, 1.0 - sqr_f32(sin_theta_t)));

    let etap = sqrt(max(0.0, b.eta * b.eta - sqr_f32(sin_theta_o))) / cos_theta_o;
    let sin_gamma_t = b.h / etap;
    let cos_gamma_t = sqrt(max(0.0, 1.0 - sqr_f32(sin_gamma_t)));

    let t = exp(-b.sigma_a * (2.0 * cos_gamma_t / cos_theta_t));
    let ap = hair_ap(cos_theta_o, b.eta, b.h, t);

    var sum_y = 0.0;
    for (var i = 0u; i <= HAIR_P_MAX; i = i + 1u) {
        sum_y += sampled_spectrum_average(ap[i]);
    }
    var ap_pdf = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0u; i <= HAIR_P_MAX; i = i + 1u) {
        ap_pdf[i] = select(0.0, max(sampled_spectrum_average(ap[i]) / sum_y, 0.0), sum_y > 0.0);
    }
    return ap_pdf;
}

// HairBxDF::ComputeThetaop as (sin, cos).
fn hair_compute_theta_op(
    b: HairBxdfState,
    p: u32,
    sin_theta_o: f32,
    cos_theta_o: f32,
) -> vec2<f32> {
    switch p {
        case 0u: {
            return vec2<f32>(
                sin_theta_o * b.cos2k_alpha[1] - cos_theta_o * b.sin2k_alpha[1],
                cos_theta_o * b.cos2k_alpha[1] + sin_theta_o * b.sin2k_alpha[1],
            );
        }
        case 1u: {
            return vec2<f32>(
                sin_theta_o * b.cos2k_alpha[0] + cos_theta_o * b.sin2k_alpha[0],
                cos_theta_o * b.cos2k_alpha[0] - sin_theta_o * b.sin2k_alpha[0],
            );
        }
        case 2u: {
            return vec2<f32>(
                sin_theta_o * b.cos2k_alpha[2] + cos_theta_o * b.sin2k_alpha[2],
                cos_theta_o * b.cos2k_alpha[2] - sin_theta_o * b.sin2k_alpha[2],
            );
        }
        default: {
            return vec2<f32>(sin_theta_o, cos_theta_o);
        }
    }
}

// sigma_a helpers used by evaluate_attributes.

// HairBxDF::SigmaAFromReflectance with beta_n.
fn hair_sigma_a_from_reflectance(c: vec4<f32>, beta_n: f32) -> vec4<f32> {
    let denom = 5.969 - 0.215 * beta_n + 2.532 * sqr_f32(beta_n)
        - 10.73 * pow(beta_n, 3.0) + 5.574 * pow(beta_n, 4.0)
        + 0.245 * pow(beta_n, 5.0);
    return sqr_vec4(log(c) / denom);
}

// HairBxDF::SigmaAFromConcentration: the v4 RGB constants become an
// sRGB-unbounded spectrum at the active wavelengths, matching the CPU
// Spectrum::from_rgb_unbounded(...).sample(lambda) path.
fn hair_sigma_a_from_pigment(eumelanin: f32, pheomelanin: f32, lambda: vec4<f32>) -> vec4<f32> {
    let ce = max(eumelanin, 0.0);
    let cp = max(pheomelanin, 0.0);
    let rgb = vec3<f32>(
        0.419 * ce + 0.187 * cp,
        0.697 * ce + 0.4 * cp,
        1.37 * ce + 1.05 * cp,
    );
    // color space 0 is sRGB (scene/texture.rs color_space_id).
    return rgb_to_unbounded_spectrum4(rgb, lambda, 0u);
}

// HairBxDF::Mp with the I0 series and LogI0 approximation from
// util/math (pbrt-v4 math.h I0 / LogI0).
fn hair_mp(cos_theta_i: f32, cos_theta_o: f32, sin_theta_i: f32, sin_theta_o: f32, v: f32) -> f32 {
    let LN2: f32 = 0.6931471805599453;

    let a = cos_theta_i * cos_theta_o / v;
    let b = sin_theta_i * sin_theta_o / v;
    let mp_value = select(
        exp(-b) * hair_i0(a) / (sinh(1.0 / v) * 2.0 * v),
        exp(hair_log_i0(a) - b - 1.0 / v + LN2 + log(0.5 / v)),
        v <= 0.1,
    );
    return max(mp_value, 0.0);
}

// HairBxDF::Ap: F, TT, TRT, ..., residual.
fn hair_ap(cos_theta_o: f32, eta: f32, h: f32, t: vec4<f32>) -> array<vec4<f32>, 4> {
    var ap = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    let cos_gamma_o = sqrt(max(0.0, 1.0 - h * h));
    let cos_theta = cos_theta_o * cos_gamma_o;
    let f = fr_dielectric(cos_theta, eta);

    ap[0] = vec4<f32>(f);
    ap[1] = sqr_vec4(vec4<f32>(1.0 - f)) * t;
    for (var p = 2u; p < HAIR_P_MAX; p = p + 1u) {
        ap[p] = ap[p - 1u] * t * f;
    }
    let one_minus_tf = vec4<f32>(1.0) - t * f;
    // SampledSpectrum::IsBlack() gates the residual lane in the CPU code.
    if (dot(one_minus_tf, one_minus_tf) > 0.0) {
        ap[HAIR_P_MAX] = ap[HAIR_P_MAX - 1u] * f * t / one_minus_tf;
    }
    return ap;
}

// HairBxDF::phi(p).
fn hair_phi(p: u32, gamma_o: f32, gamma_t: f32) -> f32 {
    return 2.0 * f32(p) * gamma_t - 2.0 * gamma_o + f32(p) * PI;
}

fn hair_logistic(x_in: f32, s: f32) -> f32 {
    let x = abs(x_in);
    return exp(-x / s) / (s * sqr_f32(1.0 + exp(-x / s)));
}

fn hair_logistic_cdf(x: f32, s: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x / s));
}

fn hair_trimmed_logistic(x: f32, s: f32, a: f32, b: f32) -> f32 {
    return hair_logistic(x, s) / (hair_logistic_cdf(b, s) - hair_logistic_cdf(a, s));
}

fn hair_np(phi: f32, p: u32, s: f32, gamma_o: f32, gamma_t: f32) -> f32 {
    var dphi = phi - hair_phi(p, gamma_o, gamma_t);
    while (dphi > PI) {
        dphi -= 2.0 * PI;
    }
    while (dphi < -PI) {
        dphi += 2.0 * PI;
    }
    return max(hair_trimmed_logistic(dphi, s, -PI, PI), 0.0);
}

fn hair_sample_trimmed_logistic(u: f32, s: f32, a: f32, b: f32) -> f32 {
    let k = hair_logistic_cdf(b, s) - hair_logistic_cdf(a, s);
    let x = -s * log(1.0 / (u * k + hair_logistic_cdf(a, s)) - 1.0);
    return clamp(x, a, b);
}

// utils/math I0: the 10-term power series of I0(x) for small arguments.
fn hair_i0(x: f32) -> f32 {
    var value = 0.0;
    var x2i = 1.0;
    var factorial = 1.0;
    var power = 1.0;
    for (var i = 0; i < 10; i = i + 1) {
        if (i > 1) {
            factorial *= f32(i);
        }
        value += x2i / (power * factorial * factorial);
        x2i *= x * x;
        power *= 4.0;
    }
    return value;
}

fn hair_log_i0(x: f32) -> f32 {
    if (x > 12.0) {
        return x + 0.5 * (-log(2.0 * PI) + log(1.0 / x) + 1.0 / (8.0 * x));
    }
    return log(hair_i0(x));
}

// u32 bit deinterleave of pbrt-v4 compact1by1. WGSL has no u64; in the CPU
// float builds the demux products fit 24 mantissa bits anyway, so the u32
// product carries the same information (the v4 float build's uint64_t
// multiplication also starts from an f32 mantissa).
fn hair_compact1_by1(x_in: u32) -> u32 {
    var x = x_in & 0x55555555u;
    x = (x ^ (x >> 1u)) & 0x33333333u;
    x = (x ^ (x >> 2u)) & 0x0f0f0f0fu;
    x = (x ^ (x >> 4u)) & 0x00ff00ffu;
    x = (x ^ (x >> 8u)) & 0x0000ffffu;
    return x;
}

fn hair_demux_float(f: f32) -> vec2<f32> {
    let v = u32(f * 4294967296.0);
    let bits = vec2<f32>(
        f32(hair_compact1_by1(v)) / 65536.0,
        f32(hair_compact1_by1(v >> 1u)) / 65536.0,
    );
    return bits;
}

fn hair_safe_asin(x: f32) -> f32 {
    return asin(clamp(x, -1.0, 1.0));
}

fn sqr_f32(x: f32) -> f32 {
    return x * x;
}

fn sqr_vec4(x: vec4<f32>) -> vec4<f32> {
    return x * x;
}

fn radians_f32(x: f32) -> f32 {
    return x * (PI / 180.0);
}
