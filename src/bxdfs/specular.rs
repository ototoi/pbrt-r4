use crate::base::bxdf::*;
use crate::util::base::*;
use crate::util::distribution::{MicrofacetDistribution, TrowbridgeReitzDistribution};
use crate::util::scattering::{abs_cos_theta, cos_theta, reflect, same_hemisphere};
use crate::util::spectrum::*;

// -----------------------------------------------------------------------
// SpecularReflectionBxDF
// -----------------------------------------------------------------------
//
// v4 material factories use the concrete conductor and dielectric variants.
#[derive(Debug, Clone, Copy)]
pub struct SpecularReflectionBxDF {
    r: SampledSpectrum,
}

impl SpecularReflectionBxDF {
    pub fn new(r: SampledSpectrum) -> Self {
        SpecularReflectionBxDF { r }
    }

    pub fn f(&self, _wo: &Vector3f, _wi: &Vector3f, _mode: TransportMode) -> SampledSpectrum {
        SampledSpectrum::zero()
    }

    pub fn sample_f(
        &self,
        wo: &Vector3f,
        _uc: Float,
        _u: &Point2f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if self.r.is_black() || wo.z == 0.0 {
            return None;
        }
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            return None;
        }
        let wi = Vector3f::new(-wo.x, -wo.y, wo.z);
        Some(BSDFSample::new(
            self.r / abs_cos_theta(&wi),
            wi,
            1.0,
            BXDF_REFLECTION | BXDF_SPECULAR,
            1.0,
            false,
        ))
    }

    pub fn pdf(
        &self,
        _wo: &Vector3f,
        _wi: &Vector3f,
        _mode: TransportMode,
        _sample_flags: BxDFReflTransFlags,
    ) -> Float {
        0.0
    }

    pub fn flags(&self) -> BxDFFlags {
        if self.r.is_black() {
            BXDF_UNSET
        } else {
            BXDF_REFLECTION | BXDF_SPECULAR
        }
    }

    pub fn regularize(&mut self) {}
}

// -----------------------------------------------------------------------
// DielectricBxDF — direct translation of pbrt-v4 `DielectricBxDF`
// (`bxdfs.h:167` and `bxdfs.cpp:77`).
// -----------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct DielectricBxDF {
    eta: Float,
    mf_distrib: TrowbridgeReitzDistribution,
}

impl DielectricBxDF {
    pub fn new(eta: Float, mf_distrib: TrowbridgeReitzDistribution) -> Self {
        DielectricBxDF { eta, mf_distrib }
    }

    pub fn flags(&self) -> BxDFFlags {
        let base = if self.eta == 1.0 {
            BXDF_TRANSMISSION
        } else {
            BXDF_REFLECTION | BXDF_TRANSMISSION
        };
        let smooth = if self.mf_distrib.effectively_smooth() {
            BXDF_SPECULAR
        } else {
            BXDF_GLOSSY
        };
        base | smooth
    }

    pub fn f(&self, wo: &Vector3f, wi: &Vector3f, mode: TransportMode) -> SampledSpectrum {
        if self.eta == 1.0 || self.mf_distrib.effectively_smooth() {
            return SampledSpectrum::zero();
        }
        // Evaluate rough dielectric BSDF.
        let cos_theta_o = cos_theta(wo);
        let cos_theta_i = cos_theta(wi);
        let reflect = cos_theta_i * cos_theta_o > 0.0;
        let mut etap = 1.0;
        if !reflect {
            etap = if cos_theta_o > 0.0 {
                self.eta
            } else {
                1.0 / self.eta
            };
        }
        let mut wm = *wi * etap + *wo;
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 || wm.length_squared() == 0.0 {
            return SampledSpectrum::zero();
        }
        wm = face_forward_z(wm.normalize());

        // Discard backfacing microfacets.
        if Vector3f::dot(&wm, wi) * cos_theta_i < 0.0 || Vector3f::dot(&wm, wo) * cos_theta_o < 0.0
        {
            return SampledSpectrum::zero();
        }

        let fr = fr_dielectric(Vector3f::dot(wo, &wm), self.eta);
        if reflect {
            SampledSpectrum::new(
                self.mf_distrib.d(&wm) * self.mf_distrib.g(wo, wi) * fr
                    / Float::abs(4.0 * cos_theta_i * cos_theta_o),
            )
        } else {
            let denom = sqr(Vector3f::dot(wi, &wm) + Vector3f::dot(wo, &wm) / etap)
                * cos_theta_i
                * cos_theta_o;
            let mut ft = self.mf_distrib.d(&wm)
                * (1.0 - fr)
                * self.mf_distrib.g(wo, wi)
                * Float::abs(Vector3f::dot(wi, &wm) * Vector3f::dot(wo, &wm) / denom);
            if mode == TransportMode::Radiance {
                ft /= etap * etap;
            }
            SampledSpectrum::new(ft)
        }
    }

    pub fn sample_f(
        &self,
        wo: &Vector3f,
        uc: Float,
        u: &Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if self.eta == 1.0 || self.mf_distrib.effectively_smooth() {
            // Sample perfectly specular dielectric BSDF.
            let r = fr_dielectric(cos_theta(wo), self.eta);
            let t = 1.0 - r;
            let mut pr = r;
            let mut pt = t;
            if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
                pr = 0.0;
            }
            if sample_flags & BXDF_REFL_TRANS_TRANSMISSION == 0 {
                pt = 0.0;
            }
            if pr == 0.0 && pt == 0.0 {
                return None;
            }

            if uc < pr / (pr + pt) {
                // Sample perfect specular dielectric BRDF.
                let wi = Vector3f::new(-wo.x, -wo.y, wo.z);
                let fr = SampledSpectrum::new(r / abs_cos_theta(&wi));
                Some(BSDFSample::new(
                    fr,
                    wi,
                    pr / (pr + pt),
                    BXDF_REFLECTION | BXDF_SPECULAR,
                    1.0,
                    false,
                ))
            } else {
                // Sample perfect specular dielectric BTDF.
                let n = Vector3f::new(0.0, 0.0, 1.0);
                let (wi, etap) = refract_v4(wo, &n, self.eta)?;
                let mut ft = SampledSpectrum::new(t / abs_cos_theta(&wi));
                if mode == TransportMode::Radiance {
                    ft /= etap * etap;
                }
                Some(BSDFSample::new(
                    ft,
                    wi,
                    pt / (pr + pt),
                    BXDF_TRANSMISSION | BXDF_SPECULAR,
                    etap,
                    false,
                ))
            }
        } else {
            // Sample rough dielectric BSDF.
            let wm = self.mf_distrib.sample_wh(wo, &Vector2f::new(u.x, u.y));
            let r = fr_dielectric(Vector3f::dot(wo, &wm), self.eta);
            let t = 1.0 - r;
            let mut pr = r;
            let mut pt = t;
            if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
                pr = 0.0;
            }
            if sample_flags & BXDF_REFL_TRANS_TRANSMISSION == 0 {
                pt = 0.0;
            }
            if pr == 0.0 && pt == 0.0 {
                return None;
            }

            if uc < pr / (pr + pt) {
                // Sample reflection at rough dielectric interface.
                let wi = reflect(wo, &wm);
                if !same_hemisphere(wo, &wi) {
                    return None;
                }
                let pdf = self.mf_distrib.pdf(wo, &wm) / (4.0 * Vector3f::abs_dot(wo, &wm)) * pr
                    / (pr + pt);
                let f = SampledSpectrum::new(
                    self.mf_distrib.d(&wm) * self.mf_distrib.g(wo, &wi) * r
                        / (4.0 * cos_theta(&wi) * cos_theta(wo)),
                );
                Some(BSDFSample::new(
                    f,
                    wi,
                    pdf,
                    BXDF_REFLECTION | BXDF_GLOSSY,
                    1.0,
                    false,
                ))
            } else {
                // Sample transmission at rough dielectric interface.
                let (wi, etap) = match refract_v4(wo, &wm, self.eta) {
                    Some(p) => p,
                    None => return None,
                };
                if same_hemisphere(wo, &wi) || wi.z == 0.0 {
                    return None;
                }
                let denom = sqr(Vector3f::dot(&wi, &wm) + Vector3f::dot(wo, &wm) / etap);
                let dwm_dwi = Vector3f::abs_dot(&wi, &wm) / denom;
                let pdf = self.mf_distrib.pdf(wo, &wm) * dwm_dwi * pt / (pr + pt);
                let mut ft = t
                    * self.mf_distrib.d(&wm)
                    * self.mf_distrib.g(wo, &wi)
                    * Float::abs(
                        Vector3f::dot(&wi, &wm) * Vector3f::dot(wo, &wm)
                            / (cos_theta(&wi) * cos_theta(wo) * denom),
                    );
                if mode == TransportMode::Radiance {
                    ft /= etap * etap;
                }
                Some(BSDFSample::new(
                    SampledSpectrum::new(ft),
                    wi,
                    pdf,
                    BXDF_TRANSMISSION | BXDF_GLOSSY,
                    etap,
                    false,
                ))
            }
        }
    }

    pub fn pdf(
        &self,
        wo: &Vector3f,
        wi: &Vector3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if self.eta == 1.0 || self.mf_distrib.effectively_smooth() {
            return 0.0;
        }
        // Evaluate sampling PDF of rough dielectric BSDF.
        let cos_theta_o = cos_theta(wo);
        let cos_theta_i = cos_theta(wi);
        let reflect = cos_theta_i * cos_theta_o > 0.0;
        let mut etap = 1.0;
        if !reflect {
            etap = if cos_theta_o > 0.0 {
                self.eta
            } else {
                1.0 / self.eta
            };
        }
        let mut wm = *wi * etap + *wo;
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 || wm.length_squared() == 0.0 {
            return 0.0;
        }
        wm = face_forward_z(wm.normalize());

        // Discard backfacing microfacets.
        if Vector3f::dot(&wm, wi) * cos_theta_i < 0.0 || Vector3f::dot(&wm, wo) * cos_theta_o < 0.0
        {
            return 0.0;
        }

        let r = fr_dielectric(Vector3f::dot(wo, &wm), self.eta);
        let t = 1.0 - r;
        let mut pr = r;
        let mut pt = t;
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            pr = 0.0;
        }
        if sample_flags & BXDF_REFL_TRANS_TRANSMISSION == 0 {
            pt = 0.0;
        }
        if pr == 0.0 && pt == 0.0 {
            return 0.0;
        }

        if reflect {
            self.mf_distrib.pdf(wo, &wm) / (4.0 * Vector3f::abs_dot(wo, &wm)) * pr / (pr + pt)
        } else {
            let denom = sqr(Vector3f::dot(wi, &wm) + Vector3f::dot(wo, &wm) / etap);
            let dwm_dwi = Vector3f::abs_dot(wi, &wm) / denom;
            self.mf_distrib.pdf(wo, &wm) * dwm_dwi * pt / (pr + pt)
        }
    }

    pub fn regularize(&mut self) {
        self.mf_distrib.regularize();
    }
}

// -----------------------------------------------------------------------
// ThinDielectricBxDF — direct translation of pbrt-v4
// `ThinDielectricBxDF` (`bxdfs.h:208`).
// -----------------------------------------------------------------------
#[derive(Debug, Clone, Copy)]
pub struct ThinDielectricBxDF {
    eta: Float,
}

impl ThinDielectricBxDF {
    pub fn new(eta: Float) -> Self {
        Self { eta }
    }

    pub fn eta(&self) -> Float {
        self.eta
    }

    pub fn f(&self, _wo: &Vector3f, _wi: &Vector3f, _mode: TransportMode) -> SampledSpectrum {
        SampledSpectrum::zero()
    }

    pub fn sample_f(
        &self,
        wo: &Vector3f,
        uc: Float,
        _u: &Point2f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        let mut r = fr_dielectric(abs_cos_theta(wo), self.eta);
        let mut t = 1.0 - r;
        // Compute R and T accounting for scattering between interfaces.
        if r < 1.0 {
            r += sqr(t) * r / (1.0 - sqr(r));
            t = 1.0 - r;
        }

        let mut pr = r;
        let mut pt = t;
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            pr = 0.0;
        }
        if sample_flags & BXDF_REFL_TRANS_TRANSMISSION == 0 {
            pt = 0.0;
        }
        if pr == 0.0 && pt == 0.0 {
            return None;
        }

        if uc < pr / (pr + pt) {
            // Sample perfect specular dielectric BRDF.
            let wi = Vector3f::new(-wo.x, -wo.y, wo.z);
            let fr = SampledSpectrum::new(r / abs_cos_theta(&wi));
            Some(BSDFSample::new(
                fr,
                wi,
                pr / (pr + pt),
                BXDF_REFLECTION | BXDF_SPECULAR,
                1.0,
                false,
            ))
        } else {
            // Sample perfect specular transmission at thin dielectric interface.
            let wi = -*wo;
            let ft = SampledSpectrum::new(t / abs_cos_theta(&wi));
            Some(BSDFSample::new(
                ft,
                wi,
                pt / (pr + pt),
                BXDF_TRANSMISSION | BXDF_SPECULAR,
                1.0,
                false,
            ))
        }
    }

    pub fn pdf(
        &self,
        _wo: &Vector3f,
        _wi: &Vector3f,
        _mode: TransportMode,
        _sample_flags: BxDFReflTransFlags,
    ) -> Float {
        0.0
    }

    pub fn flags(&self) -> BxDFFlags {
        BXDF_REFLECTION | BXDF_TRANSMISSION | BXDF_SPECULAR
    }

    pub fn regularize(&mut self) {}
}

// -----------------------------------------------------------------------
// ConductorBxDF — direct translation of pbrt-v4 `ConductorBxDF`
// (`bxdfs.h:279`).
// -----------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ConductorBxDF {
    mf_distrib: TrowbridgeReitzDistribution,
    eta: SampledSpectrum,
    k: SampledSpectrum,
}

impl ConductorBxDF {
    pub fn new(
        mf_distrib: TrowbridgeReitzDistribution,
        eta: SampledSpectrum,
        k: SampledSpectrum,
    ) -> Self {
        ConductorBxDF { mf_distrib, eta, k }
    }

    pub fn flags(&self) -> BxDFFlags {
        if self.mf_distrib.effectively_smooth() {
            BXDF_REFLECTION | BXDF_SPECULAR
        } else {
            BXDF_REFLECTION | BXDF_GLOSSY
        }
    }

    pub fn f(&self, wo: &Vector3f, wi: &Vector3f, _mode: TransportMode) -> SampledSpectrum {
        if !same_hemisphere(wo, wi) {
            return SampledSpectrum::zero();
        }
        if self.mf_distrib.effectively_smooth() {
            return SampledSpectrum::zero();
        }
        // Evaluate rough conductor BRDF.
        let cos_theta_o = abs_cos_theta(wo);
        let cos_theta_i = abs_cos_theta(wi);
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 {
            return SampledSpectrum::zero();
        }
        let mut wm = *wi + *wo;
        if wm.length_squared() == 0.0 {
            return SampledSpectrum::zero();
        }
        wm = wm.normalize();

        let f = fr_complex(Vector3f::abs_dot(wo, &wm), &self.eta, &self.k);
        self.mf_distrib.d(&wm) * f * self.mf_distrib.g(wo, wi) / (4.0 * cos_theta_i * cos_theta_o)
    }

    pub fn sample_f(
        &self,
        wo: &Vector3f,
        _uc: Float,
        u: &Point2f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            return None;
        }
        if self.mf_distrib.effectively_smooth() {
            // Sample perfect specular conductor BRDF.
            let wi = Vector3f::new(-wo.x, -wo.y, wo.z);
            let f = fr_complex(abs_cos_theta(&wi), &self.eta, &self.k) / abs_cos_theta(&wi);
            return Some(BSDFSample::new(
                f,
                wi,
                1.0,
                BXDF_REFLECTION | BXDF_SPECULAR,
                1.0,
                false,
            ));
        }
        // Sample rough conductor BRDF.
        if wo.z == 0.0 {
            return None;
        }
        let wm = self.mf_distrib.sample_wh(wo, &Vector2f::new(u.x, u.y));
        let wi = reflect(wo, &wm);
        if !same_hemisphere(wo, &wi) {
            return None;
        }

        let pdf = self.mf_distrib.pdf(wo, &wm) / (4.0 * Vector3f::abs_dot(wo, &wm));

        let cos_theta_o = abs_cos_theta(wo);
        let cos_theta_i = abs_cos_theta(&wi);
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 {
            return None;
        }
        let f_factor = fr_complex(Vector3f::abs_dot(wo, &wm), &self.eta, &self.k);
        let f = self.mf_distrib.d(&wm) * f_factor * self.mf_distrib.g(wo, &wi)
            / (4.0 * cos_theta_i * cos_theta_o);
        Some(BSDFSample::new(
            f,
            wi,
            pdf,
            BXDF_REFLECTION | BXDF_GLOSSY,
            1.0,
            false,
        ))
    }

    pub fn pdf(
        &self,
        wo: &Vector3f,
        wi: &Vector3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            return 0.0;
        }
        if !same_hemisphere(wo, wi) {
            return 0.0;
        }
        if self.mf_distrib.effectively_smooth() {
            return 0.0;
        }
        let mut wm = *wo + *wi;
        if wm.length_squared() == 0.0 {
            return 0.0;
        }
        wm = face_forward_z(wm.normalize());
        self.mf_distrib.pdf(wo, &wm) / (4.0 * Vector3f::abs_dot(wo, &wm))
    }

    pub fn regularize(&mut self) {
        self.mf_distrib.regularize();
    }
}

// -----------------------------------------------------------------------
// Local helpers (all v4-verbatim).
// -----------------------------------------------------------------------

#[inline]
pub fn sqr(v: Float) -> Float {
    v * v
}

#[inline]
fn face_forward_z(v: Vector3f) -> Vector3f {
    if v.z < 0.0 {
        -v
    } else {
        v
    }
}

/// pbrt-v4 `FrDielectric(cosTheta_i, eta)` from `util/scattering.h`.
pub fn fr_dielectric(mut cos_theta_i: Float, mut eta: Float) -> Float {
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

/// pbrt-v4 `FrComplex(cosTheta_i, eta, k)` over a `SampledSpectrum`
/// pair — computes Fresnel reflectance for a conductor per λ.
pub fn fr_complex(
    cos_theta_i: Float,
    eta: &SampledSpectrum,
    k: &SampledSpectrum,
) -> SampledSpectrum {
    let mut result = SampledSpectrum::zero();
    for i in 0..SampledSpectrum::N_SAMPLES {
        result[i] = fr_complex_scalar(cos_theta_i, eta[i], k[i]);
    }
    result
}

fn fr_complex_scalar(cos_theta_i: Float, eta_r: Float, eta_i: Float) -> Float {
    let cos_theta_i = Float::clamp(cos_theta_i, 0.0, 1.0);
    // Complex Snell's law: sin2Theta_t = sin2Theta_i / eta^2.
    let sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i;
    // eta = eta_r + i * eta_i; eta^2 = (eta_r^2 - eta_i^2) + i (2 eta_r eta_i).
    let eta2_r = eta_r * eta_r - eta_i * eta_i;
    let eta2_i = 2.0 * eta_r * eta_i;
    // sin2Theta_t (complex) = sin2_theta_i / (eta2_r + i eta2_i).
    let denom = eta2_r * eta2_r + eta2_i * eta2_i;
    let sin2_theta_t_r = sin2_theta_i * eta2_r / denom;
    let sin2_theta_t_i = -sin2_theta_i * eta2_i / denom;
    // cosTheta_t = sqrt(1 - sin2Theta_t).
    let one_minus_r = 1.0 - sin2_theta_t_r;
    let one_minus_i = -sin2_theta_t_i;
    let (cos_t_r, cos_t_i) = complex_sqrt(one_minus_r, one_minus_i);

    // r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t).
    let eta_cos_r = eta_r * cos_theta_i;
    let eta_cos_i = eta_i * cos_theta_i;
    let num_parl_r = eta_cos_r - cos_t_r;
    let num_parl_i = eta_cos_i - cos_t_i;
    let den_parl_r = eta_cos_r + cos_t_r;
    let den_parl_i = eta_cos_i + cos_t_i;
    let r_parl_norm = complex_norm_of_div(num_parl_r, num_parl_i, den_parl_r, den_parl_i);

    // r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t).
    let eta_ct_r = eta_r * cos_t_r - eta_i * cos_t_i;
    let eta_ct_i = eta_r * cos_t_i + eta_i * cos_t_r;
    let num_perp_r = cos_theta_i - eta_ct_r;
    let num_perp_i = -eta_ct_i;
    let den_perp_r = cos_theta_i + eta_ct_r;
    let den_perp_i = eta_ct_i;
    let r_perp_norm = complex_norm_of_div(num_perp_r, num_perp_i, den_perp_r, den_perp_i);

    (r_parl_norm + r_perp_norm) * 0.5
}

#[inline]
fn complex_sqrt(re: Float, im: Float) -> (Float, Float) {
    // Principal branch sqrt(re + i im).
    let r = (re * re + im * im).sqrt();
    let a = Float::sqrt(Float::max(0.0, 0.5 * (r + re)));
    let b = Float::sqrt(Float::max(0.0, 0.5 * (r - re)));
    let b_signed = if im >= 0.0 { b } else { -b };
    (a, b_signed)
}

#[inline]
fn complex_norm_of_div(num_r: Float, num_i: Float, den_r: Float, den_i: Float) -> Float {
    let denom = den_r * den_r + den_i * den_i;
    if denom == 0.0 {
        return 0.0;
    }
    let q_r = (num_r * den_r + num_i * den_i) / denom;
    let q_i = (num_i * den_r - num_r * den_i) / denom;
    q_r * q_r + q_i * q_i
}

/// pbrt-v4 `Refract(wi, n, eta, *etap, *wi)` from `util/scattering.h`.
/// Returns `(wt, etap)` on success; `None` on total internal reflection.
fn refract_v4(wi: &Vector3f, n: &Vector3f, eta: Float) -> Option<(Vector3f, Float)> {
    let mut cos_theta_i = Vector3f::dot(n, wi);
    let mut eta = eta;
    let mut n = *n;
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
        n = -n;
    }
    let sin2_theta_i = Float::max(0.0, 1.0 - cos_theta_i * cos_theta_i);
    let sin2_theta_t = sin2_theta_i / (eta * eta);
    if sin2_theta_t >= 1.0 {
        return None;
    }
    let cos_theta_t = Float::sqrt(Float::max(0.0, 1.0 - sin2_theta_t));
    let wt = -*wi / eta + (cos_theta_i / eta - cos_theta_t) * n;
    Some((wt, eta))
}
