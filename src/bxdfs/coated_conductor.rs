use super::coated_diffuse::{rng_from_directions, rng_from_sample, sample_exponential};
use super::ConductorBxDF;
use super::DielectricBxDF;
use crate::base::bxdf::*;
use crate::media::HGPhaseFunction;
use crate::util::base::*;
use crate::util::distribution::TrowbridgeReitzDistribution;
use crate::util::sampling::power_heuristic;
use crate::util::scattering::{abs_cos_theta, same_hemisphere};
use crate::util::spectrum::*;

/// Direct translation of pbrt-v4 `CoatedConductorBxDF`
/// (`bxdfs.h:912`, `LayeredBxDF<DielectricBxDF, ConductorBxDF, true>`).
/// A rough dielectric coat sits on top of a rough conductor base.
#[derive(Debug, Clone)]
pub struct CoatedConductorBxDF {
    // Top dielectric coat parameters.
    eta: Float,
    u_roughness: Float,
    v_roughness: Float,
    // Bottom conductor parameters.
    conductor_distribution: TrowbridgeReitzDistribution,
    conductor_eta: SampledSpectrum,
    conductor_k: SampledSpectrum,
    // Medium scattering inside the coat.
    albedo: SampledSpectrum,
    g: Float,
    // Sampling controls.
    thickness: Float,
    max_depth: usize,
    n_samples: usize,
}

impl CoatedConductorBxDF {
    pub fn new(
        eta: Float,
        u_roughness: Float,
        v_roughness: Float,
        conductor_distribution: TrowbridgeReitzDistribution,
        conductor_eta: SampledSpectrum,
        conductor_k: SampledSpectrum,
        albedo: SampledSpectrum,
        g: Float,
        thickness: Float,
        max_depth: usize,
        n_samples: usize,
    ) -> Self {
        Self {
            eta,
            u_roughness: u_roughness.max(0.0),
            v_roughness: v_roughness.max(0.0),
            conductor_distribution,
            conductor_eta: conductor_eta.clamp_zero(),
            conductor_k: conductor_k.clamp_zero(),
            albedo: albedo.clamp(0.0, 1.0),
            g: g.clamp(-1.0, 1.0),
            thickness: thickness.max(0.0),
            max_depth: max_depth.max(1),
            n_samples: n_samples.max(1),
        }
    }

    fn distribution(&self) -> TrowbridgeReitzDistribution {
        TrowbridgeReitzDistribution::new(self.u_roughness, self.v_roughness, true)
    }

    fn top(&self) -> DielectricBxDF {
        DielectricBxDF::new(self.eta, self.distribution())
    }

    fn bottom(&self) -> ConductorBxDF {
        ConductorBxDF::new(
            self.conductor_distribution.clone(),
            self.conductor_eta,
            self.conductor_k,
        )
    }

    fn top_is_effectively_smooth(&self) -> bool {
        Float::max(self.u_roughness, self.v_roughness) < 1e-3
    }

    fn has_medium_scattering(&self) -> bool {
        !self.albedo.is_black()
    }

    fn tr_distance(dz: Float, w: &Vector3f) -> Float {
        if Float::abs(dz) <= Float::MIN_POSITIVE {
            1.0
        } else {
            (-Float::abs(dz / w.z)).exp()
        }
    }

    pub fn f(&self, wo: &Vector3f, wi: &Vector3f, mode: TransportMode) -> SampledSpectrum {
        if wo.z == 0.0 || wi.z == 0.0 {
            return SampledSpectrum::zero();
        }

        let mut wo = *wo;
        let mut wi = *wi;
        if wo.z < 0.0 {
            wo = -wo;
            wi = -wi;
        }
        if !same_hemisphere(&wo, &wi) {
            return SampledSpectrum::zero();
        }

        let top = self.top();
        let mut f = top.f(&wo, &wi, mode) * self.n_samples as Float;
        let mut rng = rng_from_directions(&wo, &wi);
        let bottom = self.bottom();
        let phase = HGPhaseFunction::new(self.g);

        for _ in 0..self.n_samples {
            let wos = match top.sample_f(
                &wo,
                rng.uniform_float(),
                &Point2f::new(rng.uniform_float(), rng.uniform_float()),
                mode,
                BXDF_REFL_TRANS_TRANSMISSION,
            ) {
                Some(bs) if !bs.f.is_black() && bs.pdf > 0.0 && bs.wi.z != 0.0 => bs,
                _ => continue,
            };
            let wis = match top.sample_f(
                &wi,
                rng.uniform_float(),
                &Point2f::new(rng.uniform_float(), rng.uniform_float()),
                !mode,
                BXDF_REFL_TRANS_TRANSMISSION,
            ) {
                Some(bs) if !bs.f.is_black() && bs.pdf > 0.0 && bs.wi.z != 0.0 => bs,
                _ => continue,
            };

            let mut beta = wos.f * (abs_cos_theta(&wos.wi) / wos.pdf);
            let mut z = self.thickness;
            let mut w = wos.wi;

            for depth in 0..self.max_depth {
                let rr_beta = beta.max_component_value();
                if depth > 3 && rr_beta < 0.25 {
                    let q = Float::max(0.0, 1.0 - rr_beta);
                    if rng.uniform_float() < q {
                        break;
                    }
                    beta /= 1.0 - q;
                }

                if !self.has_medium_scattering() {
                    z = if z == self.thickness {
                        0.0
                    } else {
                        self.thickness
                    };
                    beta *= Self::tr_distance(self.thickness, &w);
                } else {
                    let sigma_t = 1.0;
                    let dz = sample_exponential(rng.uniform_float(), sigma_t / abs_cos_theta(&w));
                    let zp = if w.z > 0.0 { z + dz } else { z - dz };
                    if z == zp {
                        continue;
                    }
                    if 0.0 < zp && zp < self.thickness {
                        let phase_pdf = phase.p(&(-w), &(-wis.wi));
                        let wt = if self.top_is_effectively_smooth() {
                            1.0
                        } else {
                            power_heuristic(1, wis.pdf, 1, phase_pdf)
                        };
                        f += beta
                            * self.albedo
                            * phase_pdf
                            * wt
                            * Self::tr_distance(zp - self.thickness, &wis.wi)
                            * wis.f
                            / wis.pdf;

                        let u_phase = Point2f::new(rng.uniform_float(), rng.uniform_float());
                        let (phase_p, phase_wi) = phase.sample_p(&(-w), &u_phase);
                        if phase_p == 0.0 || phase_wi.z == 0.0 {
                            continue;
                        }
                        beta *= self.albedo;
                        w = phase_wi;
                        z = zp;

                        if w.z > 0.0 && !self.top_is_effectively_smooth() {
                            let f_exit = top.f(&(-w), &wi, mode);
                            if !f_exit.is_black() {
                                let exit_pdf =
                                    top.pdf(&(-w), &wi, mode, BXDF_REFL_TRANS_TRANSMISSION);
                                let wt = power_heuristic(1, phase_p, 1, exit_pdf);
                                f += beta
                                    * Self::tr_distance(zp - self.thickness, &phase_wi)
                                    * f_exit
                                    * wt;
                            }
                        }

                        continue;
                    }

                    z = zp.clamp(0.0, self.thickness);
                }

                if z == self.thickness {
                    let bs = match top.sample_f(
                        &(-w),
                        rng.uniform_float(),
                        &Point2f::new(rng.uniform_float(), rng.uniform_float()),
                        mode,
                        BXDF_REFL_TRANS_REFLECTION,
                    ) {
                        Some(bs) if !bs.f.is_black() && bs.pdf > 0.0 && bs.wi.z != 0.0 => bs,
                        _ => break,
                    };
                    beta *= bs.f * (abs_cos_theta(&bs.wi) / bs.pdf);
                    w = bs.wi;
                } else {
                    let wt = if self.top_is_effectively_smooth() {
                        1.0
                    } else {
                        power_heuristic(
                            1,
                            wis.pdf,
                            1,
                            bottom.pdf(&(-w), &(-wis.wi), mode, BXDF_REFL_TRANS_REFLECTION),
                        )
                    };
                    f += beta
                        * bottom.f(&(-w), &(-wis.wi), mode)
                        * abs_cos_theta(&wis.wi)
                        * wt
                        * Self::tr_distance(self.thickness, &wis.wi)
                        * wis.f
                        / wis.pdf;

                    let bs = match bottom.sample_f(
                        &(-w),
                        rng.uniform_float(),
                        &Point2f::new(rng.uniform_float(), rng.uniform_float()),
                        mode,
                        BXDF_REFL_TRANS_REFLECTION,
                    ) {
                        Some(bs) if !bs.f.is_black() && bs.pdf > 0.0 && bs.wi.z != 0.0 => bs,
                        _ => break,
                    };
                    beta *= bs.f * (abs_cos_theta(&bs.wi) / bs.pdf);
                    w = bs.wi;

                    if !self.top_is_effectively_smooth() {
                        let f_exit = top.f(&(-w), &wi, mode);
                        if !f_exit.is_black() {
                            let exit_pdf = top.pdf(&(-w), &wi, mode, BXDF_REFL_TRANS_TRANSMISSION);
                            let wt = power_heuristic(1, bs.pdf, 1, exit_pdf);
                            f += beta * Self::tr_distance(self.thickness, &bs.wi) * f_exit * wt;
                        }
                    }
                }
            }
        }

        f / self.n_samples as Float
    }

    pub fn sample_f(
        &self,
        wo: &Vector3f,
        uc: Float,
        u: &Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if wo.z == 0.0 || sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            return None;
        }

        let mut wo = *wo;
        let mut flip_wi = false;
        if wo.z < 0.0 {
            wo = -wo;
            flip_wi = true;
        }

        let top = self.top();
        let mut bs = top.sample_f(&wo, uc, u, mode, BXDF_REFL_TRANS_ALL)?;
        if bs.f.is_black() || bs.pdf == 0.0 || bs.wi.z == 0.0 {
            return None;
        }
        if bs.is_reflection() {
            if flip_wi {
                bs.wi = -bs.wi;
            }
            bs.pdf_is_proportional = true;
            return Some(bs);
        }

        let mut w = bs.wi;
        let mut specular_path = bs.is_specular();
        let mut rng = rng_from_sample(&wo, uc, u);
        let mut f = bs.f * abs_cos_theta(&bs.wi);
        let mut pdf = bs.pdf;
        let mut z = self.thickness;
        let bottom = self.bottom();
        let phase = HGPhaseFunction::new(self.g);

        for depth in 0..self.max_depth {
            let rr_beta = f.max_component_value() / pdf;
            if depth > 3 && rr_beta < 0.25 {
                let q = Float::max(0.0, 1.0 - rr_beta);
                if rng.uniform_float() < q {
                    return None;
                }
                pdf *= 1.0 - q;
            }
            if w.z == 0.0 {
                return None;
            }

            if self.has_medium_scattering() {
                let sigma_t = 1.0;
                let dz = sample_exponential(rng.uniform_float(), sigma_t / abs_cos_theta(&w));
                let zp = if w.z > 0.0 { z + dz } else { z - dz };
                if zp == z {
                    return None;
                }
                if 0.0 < zp && zp < self.thickness {
                    let u_phase = Point2f::new(rng.uniform_float(), rng.uniform_float());
                    let (phase_p, phase_wi) = phase.sample_p(&(-w), &u_phase);
                    if phase_p == 0.0 || phase_wi.z == 0.0 {
                        return None;
                    }
                    f *= self.albedo * phase_p;
                    pdf *= phase_p;
                    specular_path = false;
                    w = phase_wi;
                    z = zp;
                    continue;
                }
                z = zp.clamp(0.0, self.thickness);
            } else {
                z = if z == self.thickness {
                    0.0
                } else {
                    self.thickness
                };
                f *= Self::tr_distance(self.thickness, &w);
            }

            let bs = if z == self.thickness {
                top.sample_f(
                    &(-w),
                    rng.uniform_float(),
                    &Point2f::new(rng.uniform_float(), rng.uniform_float()),
                    mode,
                    BXDF_REFL_TRANS_ALL,
                )
            } else {
                bottom.sample_f(
                    &(-w),
                    rng.uniform_float(),
                    &Point2f::new(rng.uniform_float(), rng.uniform_float()),
                    mode,
                    BXDF_REFL_TRANS_REFLECTION,
                )
            }?;
            if bs.f.is_black() || bs.pdf == 0.0 || bs.wi.z == 0.0 {
                return None;
            }

            f *= bs.f;
            pdf *= bs.pdf;
            specular_path &= bs.is_specular();
            w = bs.wi;

            if z == self.thickness && bs.is_transmission() {
                let mut flags = BXDF_REFLECTION;
                flags |= if specular_path {
                    BXDF_SPECULAR
                } else {
                    BXDF_GLOSSY
                };
                if flip_wi {
                    w = -w;
                }
                return Some(BSDFSample::new(f, w, pdf, flags, 1.0, true));
            }

            f *= abs_cos_theta(&bs.wi);
        }

        None
    }

    pub fn pdf(
        &self,
        wo: &Vector3f,
        wi: &Vector3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if wo.z == 0.0 || sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            return 0.0;
        }

        let mut wo = *wo;
        let mut wi = *wi;
        if wo.z < 0.0 {
            wo = -wo;
            wi = -wi;
        }
        if !same_hemisphere(&wo, &wi) {
            return 0.0;
        }

        let top = self.top();
        let mut pdf_sum =
            self.n_samples as Float * top.pdf(&wo, &wi, mode, BXDF_REFL_TRANS_REFLECTION);
        let mut rng = rng_from_directions(&wi, &wo);
        let bottom = self.bottom();

        for _ in 0..self.n_samples {
            let wos = match top.sample_f(
                &wo,
                rng.uniform_float(),
                &Point2f::new(rng.uniform_float(), rng.uniform_float()),
                mode,
                BXDF_REFL_TRANS_TRANSMISSION,
            ) {
                Some(bs) if !bs.f.is_black() && bs.pdf > 0.0 && bs.wi.z != 0.0 => bs,
                _ => continue,
            };
            let wis = match top.sample_f(
                &wi,
                rng.uniform_float(),
                &Point2f::new(rng.uniform_float(), rng.uniform_float()),
                !mode,
                BXDF_REFL_TRANS_TRANSMISSION,
            ) {
                Some(bs) if !bs.f.is_black() && bs.pdf > 0.0 && bs.wi.z != 0.0 => bs,
                _ => continue,
            };

            let rs = match bottom.sample_f(
                &(-wos.wi),
                rng.uniform_float(),
                &Point2f::new(rng.uniform_float(), rng.uniform_float()),
                mode,
                BXDF_REFL_TRANS_REFLECTION,
            ) {
                Some(bs) if !bs.f.is_black() && bs.pdf > 0.0 && bs.wi.z != 0.0 => bs,
                _ => continue,
            };

            let r_pdf = bottom.pdf(&(-wos.wi), &(-wis.wi), mode, BXDF_REFL_TRANS_REFLECTION);
            let mut wt = power_heuristic(1, wis.pdf, 1, r_pdf);
            pdf_sum += wt * r_pdf;

            let t_pdf = top.pdf(&(-rs.wi), &wi, mode, BXDF_REFL_TRANS_TRANSMISSION);
            wt = power_heuristic(1, rs.pdf, 1, t_pdf);
            pdf_sum += wt * t_pdf;
        }

        lerp(0.9, INV_4_PI, pdf_sum / self.n_samples as Float)
    }

    pub fn flags(&self) -> BxDFFlags {
        let mut flags = BXDF_REFLECTION;
        if self.top_is_effectively_smooth() && self.conductor_distribution.effectively_smooth() {
            flags |= BXDF_SPECULAR;
        } else {
            flags |= BXDF_GLOSSY;
        }
        flags
    }

    pub fn regularize(&mut self) {
        self.conductor_distribution.regularize();
    }
}
