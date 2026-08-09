use crate::base::bxdf::*;
use crate::util::base::*;
use crate::util::math::{i0, log_i0};
use crate::util::scattering::abs_cos_theta;
use crate::util::spectrum::*;

/// Direct translation of pbrt-v4 `HairBxDF` (`bxdfs.h:921`,
/// `bxdfs.cpp:275`). `sigma_a` is a `SampledSpectrum` evaluated at the
/// path's wavelengths; v4 has no dense `Spectrum` field here.

const P_MAX: usize = 3;
const SQRT_PI_OVER8: Float = 0.626657069;

#[derive(Debug, Clone, Copy)]
pub struct HairBxDF {
    h: Float,
    gamma_o: Float,
    eta: Float,
    sigma_a: SampledSpectrum,
    v: [Float; P_MAX + 1],
    s: Float,
    sin2k_alpha: [Float; 3],
    cos2k_alpha: [Float; 3],
}

impl HairBxDF {
    pub fn new(
        h: Float,
        eta: Float,
        sigma_a: SampledSpectrum,
        beta_m: Float,
        beta_n: Float,
        alpha: Float,
    ) -> Self {
        let gamma_o = safe_asin(h);
        debug_assert!((-1.0..=1.0).contains(&h));
        debug_assert!((0.0..=1.0).contains(&beta_m));
        debug_assert!((0.0..=1.0).contains(&beta_n));

        let mut v = [0.0; P_MAX + 1];
        let v0 = sqr(0.726 * beta_m + 0.812 * sqr(beta_m) + 3.7 * Float::powf(beta_m, 20.0));
        v[0] = v0;
        v[1] = 0.25 * v0;
        v[2] = 4.0 * v0;
        let v2 = v[2];
        for vi in v.iter_mut().skip(3) {
            *vi = v2;
        }

        let s = SQRT_PI_OVER8
            * (0.265 * beta_n + 1.194 * sqr(beta_n) + 5.372 * Float::powf(beta_n, 22.0));

        let sin0 = Float::sin(radians(alpha));
        let cos0 = safe_sqrt(1.0 - sqr(sin0));
        let mut sin2k_alpha = [sin0, 0.0, 0.0];
        let mut cos2k_alpha = [cos0, 0.0, 0.0];
        for i in 1..3 {
            sin2k_alpha[i] = 2.0 * cos2k_alpha[i - 1] * sin2k_alpha[i - 1];
            cos2k_alpha[i] = sqr(cos2k_alpha[i - 1]) - sqr(sin2k_alpha[i - 1]);
        }

        Self {
            h,
            gamma_o,
            eta,
            sigma_a,
            v,
            s,
            sin2k_alpha,
            cos2k_alpha,
        }
    }

    /// pbrt-v4 `HairBxDF::SigmaAFromConcentration` returns an
    /// `RGBUnboundedSpectrum` in v4. For r4 we return a SampledSpectrum
    /// over the active wavelengths so the BxDF holds it directly.
    pub fn sigma_a_from_concentration(
        eumelanin: Float,
        pheomelanin: Float,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        let ce = eumelanin.max(0.0);
        let cp = pheomelanin.max(0.0);
        // The v4 RGB sigma_a is built from per-pigment RGB constants;
        // r4 samples that RGB by mapping each lambda to the
        // sRGB-RGB-unbounded spectrum and projecting.
        let eumelanin_rgb = [0.419, 0.697, 1.37];
        let pheomelanin_rgb = [0.187, 0.4, 1.05];
        let rgb = [
            ce * eumelanin_rgb[0] + cp * pheomelanin_rgb[0],
            ce * eumelanin_rgb[1] + cp * pheomelanin_rgb[1],
            ce * eumelanin_rgb[2] + cp * pheomelanin_rgb[2],
        ];
        Spectrum::from_rgb_unbounded(&rgb).sample(lambda)
    }

    /// pbrt-v4 `HairBxDF::SigmaAFromReflectance`.
    pub fn sigma_a_from_reflectance(c: SampledSpectrum, beta_n: Float) -> SampledSpectrum {
        let denom = 5.969 - 0.215 * beta_n + 2.532 * sqr(beta_n) - 10.73 * Float::powf(beta_n, 3.0)
            + 5.574 * Float::powf(beta_n, 4.0)
            + 0.245 * Float::powf(beta_n, 5.0);
        let mut sigma_a = SampledSpectrum::zero();
        for i in 0..SampledSpectrum::N_SAMPLES {
            sigma_a[i] = sqr(Float::ln(c[i]) / denom);
        }
        sigma_a
    }

    /// pbrt-v4 `HairBxDF::f`.
    pub fn f(&self, wo: &Vector3f, wi: &Vector3f, _mode: TransportMode) -> SampledSpectrum {
        let sin_theta_o = wo.x;
        let cos_theta_o = safe_sqrt(1.0 - sqr(sin_theta_o));
        let phi_o = Float::atan2(wo.z, wo.y);

        let sin_theta_i = wi.x;
        let cos_theta_i = safe_sqrt(1.0 - sqr(sin_theta_i));
        let phi_i = Float::atan2(wi.z, wi.y);

        let sin_theta_t = sin_theta_o / self.eta;
        let cos_theta_t = safe_sqrt(1.0 - sqr(sin_theta_t));

        let etap = Float::sqrt(self.eta * self.eta - sqr(sin_theta_o)) / cos_theta_o;
        let sin_gamma_t = self.h / etap;
        let cos_gamma_t = safe_sqrt(1.0 - sqr(sin_gamma_t));
        let gamma_t = safe_asin(sin_gamma_t);

        let t = (-self.sigma_a * (2.0 * cos_gamma_t / cos_theta_t)).exp();
        let ap = ap(cos_theta_o, self.eta, self.h, t);

        let phi = phi_i - phi_o;
        let mut fsum = SampledSpectrum::zero();
        for p in 0..P_MAX {
            let (sin_theta_op, cos_theta_op) = self.compute_theta_op(p, sin_theta_o, cos_theta_o);
            let cos_theta_op = Float::abs(cos_theta_op);
            fsum += ap[p]
                * mp(
                    cos_theta_i,
                    cos_theta_op,
                    sin_theta_i,
                    sin_theta_op,
                    self.v[p],
                )
                * np(phi, p, self.s, self.gamma_o, gamma_t);
        }

        fsum += ap[P_MAX]
            * mp(
                cos_theta_i,
                cos_theta_o,
                sin_theta_i,
                sin_theta_o,
                self.v[P_MAX],
            )
            / (2.0 * PI);

        let abs_cos_theta_wi = abs_cos_theta(wi);
        if abs_cos_theta_wi > 0.0 {
            fsum /= abs_cos_theta_wi;
        }
        fsum
    }

    /// pbrt-v4 `HairBxDF::Sample_f`.
    pub fn sample_f(
        &self,
        wo: &Vector3f,
        _uc: Float,
        u: &Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if sample_flags == BXDF_REFL_TRANS_UNSET {
            return None;
        }

        let sin_theta_o = wo.x;
        let cos_theta_o = safe_sqrt(1.0 - sqr(sin_theta_o));
        let phi_o = Float::atan2(wo.z, wo.y);

        let mut uu = [demux_float(u[0]), demux_float(u[1])];
        let ap_pdf = self.compute_ap_pdf(cos_theta_o);

        let mut p = P_MAX;
        for (i, pdf) in ap_pdf.iter().enumerate() {
            if uu[0][0] < *pdf {
                p = i;
                break;
            }
            uu[0][0] -= *pdf;
        }

        let (sin_theta_op, cos_theta_op) = self.compute_theta_op(p, sin_theta_o, cos_theta_o);
        uu[1][0] = Float::max(uu[1][0], 1e-5);
        let cos_theta =
            1.0 + self.v[p] * Float::ln(uu[1][0] + (1.0 - uu[1][0]) * Float::exp(-2.0 / self.v[p]));
        let sin_theta = safe_sqrt(1.0 - sqr(cos_theta));
        let cos_phi = Float::cos(2.0 * PI * uu[1][1]);
        let sin_theta_i = -cos_theta * sin_theta_op + sin_theta * cos_phi * cos_theta_op;
        let cos_theta_i = safe_sqrt(1.0 - sqr(sin_theta_i));

        let etap = Float::sqrt(self.eta * self.eta - sqr(sin_theta_o)) / cos_theta_o;
        let sin_gamma_t = self.h / etap;
        let gamma_t = safe_asin(sin_gamma_t);
        let dphi = if p < P_MAX {
            phi(p, self.gamma_o, gamma_t) + sample_trimmed_logistic(uu[0][1], self.s, -PI, PI)
        } else {
            2.0 * PI * uu[0][1]
        };

        let phi_i = phi_o + dphi;
        let wi = Vector3f::new(
            sin_theta_i,
            cos_theta_i * Float::cos(phi_i),
            cos_theta_i * Float::sin(phi_i),
        );

        let pdf = self.pdf(wo, &wi, mode, sample_flags);
        if pdf <= 0.0 {
            return None;
        }

        Some(BSDFSample::new(
            self.f(wo, &wi, mode),
            wi,
            pdf,
            BXDF_GLOSSY | BXDF_REFLECTION | BXDF_TRANSMISSION,
            1.0,
            false,
        ))
    }

    /// pbrt-v4 `HairBxDF::PDF`.
    pub fn pdf(
        &self,
        wo: &Vector3f,
        wi: &Vector3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if sample_flags == BXDF_REFL_TRANS_UNSET {
            return 0.0;
        }

        let sin_theta_o = wo.x;
        let cos_theta_o = safe_sqrt(1.0 - sqr(sin_theta_o));
        let phi_o = Float::atan2(wo.z, wo.y);

        let sin_theta_i = wi.x;
        let cos_theta_i = safe_sqrt(1.0 - sqr(sin_theta_i));
        let phi_i = Float::atan2(wi.z, wi.y);

        let etap = Float::sqrt(self.eta * self.eta - sqr(sin_theta_o)) / cos_theta_o;
        let sin_gamma_t = self.h / etap;
        let gamma_t = safe_asin(sin_gamma_t);

        let ap_pdf = self.compute_ap_pdf(cos_theta_o);
        let phi = phi_i - phi_o;
        let mut pdf = 0.0;
        for p in 0..P_MAX {
            let (sin_theta_op, cos_theta_op) = self.compute_theta_op(p, sin_theta_o, cos_theta_o);
            let cos_theta_op = Float::abs(cos_theta_op);
            pdf += mp(
                cos_theta_i,
                cos_theta_op,
                sin_theta_i,
                sin_theta_op,
                self.v[p],
            ) * ap_pdf[p]
                * np(phi, p, self.s, self.gamma_o, gamma_t);
        }
        pdf += mp(
            cos_theta_i,
            cos_theta_o,
            sin_theta_i,
            sin_theta_o,
            self.v[P_MAX],
        ) * ap_pdf[P_MAX]
            / (2.0 * PI);
        pdf
    }

    pub fn flags(&self) -> BxDFFlags {
        BXDF_GLOSSY | BXDF_REFLECTION | BXDF_TRANSMISSION
    }

    pub fn regularize(&mut self) {}

    fn compute_ap_pdf(&self, cos_theta_o: Float) -> [Float; P_MAX + 1] {
        let sin_theta_o = safe_sqrt(1.0 - cos_theta_o * cos_theta_o);
        let sin_theta_t = sin_theta_o / self.eta;
        let cos_theta_t = safe_sqrt(1.0 - sqr(sin_theta_t));

        let etap = Float::sqrt(self.eta * self.eta - sqr(sin_theta_o)) / cos_theta_o;
        let sin_gamma_t = self.h / etap;
        let cos_gamma_t = safe_sqrt(1.0 - sqr(sin_gamma_t));

        let t = (-self.sigma_a * (2.0 * cos_gamma_t / cos_theta_t)).exp();
        let ap = ap(cos_theta_o, self.eta, self.h, t);

        let mut sum_y = 0.0;
        for s in &ap {
            sum_y += s.average();
        }
        let mut ap_pdf = [0.0 as Float; P_MAX + 1];
        for i in 0..=P_MAX {
            ap_pdf[i] = if sum_y > 0.0 {
                (ap[i].average() / sum_y).max(0.0)
            } else {
                0.0
            };
        }
        ap_pdf
    }

    fn compute_theta_op(&self, p: usize, sin_theta_o: Float, cos_theta_o: Float) -> (Float, Float) {
        match p {
            0 => (
                sin_theta_o * self.cos2k_alpha[1] - cos_theta_o * self.sin2k_alpha[1],
                cos_theta_o * self.cos2k_alpha[1] + sin_theta_o * self.sin2k_alpha[1],
            ),
            1 => (
                sin_theta_o * self.cos2k_alpha[0] + cos_theta_o * self.sin2k_alpha[0],
                cos_theta_o * self.cos2k_alpha[0] - sin_theta_o * self.sin2k_alpha[0],
            ),
            2 => (
                sin_theta_o * self.cos2k_alpha[2] + cos_theta_o * self.sin2k_alpha[2],
                cos_theta_o * self.cos2k_alpha[2] - sin_theta_o * self.sin2k_alpha[2],
            ),
            _ => (sin_theta_o, cos_theta_o),
        }
    }
}

#[inline]
fn mp(
    cos_theta_i: Float,
    cos_theta_o: Float,
    sin_theta_i: Float,
    sin_theta_o: Float,
    v: Float,
) -> Float {
    const LN2: Float = std::f32::consts::LN_2 as Float;

    let a = cos_theta_i * cos_theta_o / v;
    let b = sin_theta_i * sin_theta_o / v;
    let mp = if v <= 0.1 {
        Float::exp(log_i0(a) - b - 1.0 / v + LN2 + Float::ln(1.0 / (2.0 * v)))
    } else {
        Float::exp(-b) * i0(a) / (Float::sinh(1.0 / v) * 2.0 * v)
    };
    mp.max(0.0)
}

#[inline]
fn ap(
    cos_theta_o: Float,
    eta: Float,
    h: Float,
    t: SampledSpectrum,
) -> [SampledSpectrum; P_MAX + 1] {
    let mut ap = [SampledSpectrum::zero(); P_MAX + 1];
    let cos_gamma_o = safe_sqrt(1.0 - h * h);
    let cos_theta = cos_theta_o * cos_gamma_o;
    let f = fr_dielectric(cos_theta, eta);

    ap[0] = SampledSpectrum::new(f);
    ap[1] = sqr(1.0 - f) * t;
    for p in 2..P_MAX {
        ap[p] = ap[p - 1] * t * f;
    }
    let one_minus_tf = SampledSpectrum::one() - t * f;
    if !one_minus_tf.is_black() {
        ap[P_MAX] = ap[P_MAX - 1] * f * t / one_minus_tf;
    }
    ap
}

#[inline]
fn phi(p: usize, gamma_o: Float, gamma_t: Float) -> Float {
    2.0 * p as Float * gamma_t - 2.0 * gamma_o + p as Float * PI
}

#[inline]
fn logistic(x: Float, s: Float) -> Float {
    let x = Float::abs(x);
    Float::exp(-x / s) / (s * sqr(1.0 + Float::exp(-x / s)))
}

#[inline]
fn logistic_cdf(x: Float, s: Float) -> Float {
    1.0 / (1.0 + Float::exp(-x / s))
}

#[inline]
fn trimmed_logistic(x: Float, s: Float, a: Float, b: Float) -> Float {
    debug_assert!(a < b);
    logistic(x, s) / (logistic_cdf(b, s) - logistic_cdf(a, s))
}

#[inline]
fn np(phi_: Float, p: usize, s: Float, gamma_o: Float, gamma_t: Float) -> Float {
    let mut dphi = phi_ - phi(p, gamma_o, gamma_t);
    while dphi > PI {
        dphi -= 2.0 * PI;
    }
    while dphi < -PI {
        dphi += 2.0 * PI;
    }
    trimmed_logistic(dphi, s, -PI, PI).max(0.0)
}

#[inline]
fn sample_trimmed_logistic(u: Float, s: Float, a: Float, b: Float) -> Float {
    debug_assert!(a < b);
    let k = logistic_cdf(b, s) - logistic_cdf(a, s);
    let x = -s * Float::ln(1.0 / (u * k + logistic_cdf(a, s)) - 1.0);
    Float::clamp(x, a, b)
}

#[inline]
fn safe_asin(x: Float) -> Float {
    Float::asin(Float::clamp(x, -1.0, 1.0))
}

#[inline]
fn safe_sqrt(x: Float) -> Float {
    Float::sqrt(Float::max(0.0, x))
}

#[inline]
fn sqr(x: Float) -> Float {
    x * x
}

#[inline]
fn radians(x: Float) -> Float {
    x * (PI / 180.0)
}

#[inline]
fn compact1_by1(mut x: u32) -> u32 {
    x &= 0x5555_5555;
    x = (x ^ (x >> 1)) & 0x3333_3333;
    x = (x ^ (x >> 2)) & 0x0f0f_0f0f;
    x = (x ^ (x >> 4)) & 0x00ff_00ff;
    x = (x ^ (x >> 8)) & 0x0000_ffff;
    x
}

#[inline]
fn demux_float(f: Float) -> Point2f {
    debug_assert!((0.0..1.0).contains(&f));
    let v = (f as f64 * ((1u64 << 32) as f64)) as u64;
    let bits = [
        compact1_by1((v & 0xffff_ffff) as u32),
        compact1_by1((v >> 1) as u32),
    ];
    Point2f::new(
        bits[0] as Float / (1 << 16) as Float,
        bits[1] as Float / (1 << 16) as Float,
    )
}

fn fr_dielectric(mut cos_theta_i: Float, mut eta: Float) -> Float {
    cos_theta_i = Float::clamp(cos_theta_i, -1.0, 1.0);
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
    }

    let sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i;
    let sin2_theta_t = sin2_theta_i / (eta * eta);
    if sin2_theta_t >= 1.0 {
        return 1.0;
    }
    let cos_theta_t = Float::sqrt(Float::max(0.0, 1.0 - sin2_theta_t));
    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
    (r_parl * r_parl + r_perp * r_perp) * 0.5
}
