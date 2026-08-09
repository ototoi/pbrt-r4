use super::DielectricBxDF;
use super::DiffuseBxDF;
use crate::base::bxdf::*;
use crate::media::HGPhaseFunction;
use crate::util::base::*;
use crate::util::distribution::TrowbridgeReitzDistribution;
use crate::util::rng::RNG;
use crate::util::sampling::power_heuristic;
use crate::util::scattering::{abs_cos_theta, refract, same_hemisphere};
use crate::util::spectrum::*;

/// Direct translation of pbrt-v4 `CoatedDiffuseBxDF`
/// (`bxdfs.h:902`, `bxdfs.cpp` `LayeredBxDF<DielectricBxDF, DiffuseBxDF, true>`).
/// A rough dielectric coat sits on top of a Lambertian diffuse base; rays
/// random-walk between the two interfaces with optional participating
/// medium scattering inside the coat.
#[derive(Debug, Clone, Copy)]
pub struct CoatedDiffuseBxDF {
    reflectance: SampledSpectrum,
    albedo: SampledSpectrum,
    g: Float,
    eta: Float,
    u_roughness: Float,
    v_roughness: Float,
    thickness: Float,
    max_depth: usize,
    n_samples: usize,
}

impl CoatedDiffuseBxDF {
    pub fn new(
        reflectance: SampledSpectrum,
        albedo: SampledSpectrum,
        g: Float,
        eta: Float,
        u_roughness: Float,
        v_roughness: Float,
        thickness: Float,
        max_depth: usize,
        n_samples: usize,
    ) -> Self {
        CoatedDiffuseBxDF {
            reflectance,
            albedo: albedo.clamp(0.0, 1.0),
            g: g.clamp(-1.0, 1.0),
            eta,
            u_roughness,
            v_roughness,
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

    fn bottom(&self) -> DiffuseBxDF {
        DiffuseBxDF::new(self.reflectance)
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

    /// pbrt-v4 `LayeredBxDF::f` (specialized for the diffuse base).
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

    /// pbrt-v4 `LayeredBxDF::Sample_f`.
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

    /// pbrt-v4 `LayeredBxDF::PDF`.
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

    /// pbrt-v4 `LayeredBxDF::Flags` for `<DielectricBxDF, DiffuseBxDF, true>`.
    pub fn flags(&self) -> BxDFFlags {
        let mut flags = BXDF_REFLECTION;
        if self.top_is_effectively_smooth() {
            flags |= BXDF_SPECULAR;
        }
        if !self.reflectance.is_black() || self.has_medium_scattering() {
            flags |= BXDF_DIFFUSE;
        } else {
            flags |= BXDF_GLOSSY;
        }
        flags
    }

    pub fn regularize(&mut self) {}
}

// ---------- shared helpers reused by CoatedConductorBxDF -------------

pub fn refract_with_etap(wo: &Vector3f, n: &Vector3f, eta: Float) -> Option<(Vector3f, Float)> {
    let entering = Vector3f::dot(wo, n) > 0.0;
    let (eta_i, eta_t, nn) = if entering {
        (1.0, eta.max(1e-4), *n)
    } else {
        (eta.max(1e-4), 1.0, -*n)
    };
    let wi = refract(wo, &nn, eta_i / eta_t)?;
    Some((wi, eta_t / eta_i))
}

pub fn rng_from_directions(wo: &Vector3f, wi: &Vector3f) -> RNG {
    let sequence = pbrt_hash_seed_vector(0, wo);
    let seed = pbrt_hash_vector(wi);
    let mut rng = RNG::new();
    rng.set_sequence_with_seed(sequence, seed);
    rng
}

pub fn rng_from_sample(wo: &Vector3f, uc: Float, u: &Point2f) -> RNG {
    let sequence = pbrt_hash_seed_vector(0, wo);
    let seed = pbrt_hash_sample(uc, u);
    let mut rng = RNG::new();
    rng.set_sequence_with_seed(sequence, seed);
    rng
}

fn pbrt_hash_seed_vector(seed: i32, v: &Vector3f) -> u64 {
    let mut bytes =
        Vec::with_capacity(std::mem::size_of::<i32>() + 3 * std::mem::size_of::<Float>());
    bytes.extend_from_slice(&seed.to_ne_bytes());
    push_vector3_bytes(&mut bytes, v);
    murmur_hash_64a(&bytes, 0)
}

fn pbrt_hash_vector(v: &Vector3f) -> u64 {
    let mut bytes = Vec::with_capacity(3 * std::mem::size_of::<Float>());
    push_vector3_bytes(&mut bytes, v);
    murmur_hash_64a(&bytes, 0)
}

fn pbrt_hash_sample(uc: Float, u: &Point2f) -> u64 {
    let mut bytes = Vec::with_capacity(3 * std::mem::size_of::<Float>());
    push_float_bytes(&mut bytes, uc);
    push_float_bytes(&mut bytes, u.x);
    push_float_bytes(&mut bytes, u.y);
    murmur_hash_64a(&bytes, 0)
}

fn push_vector3_bytes(bytes: &mut Vec<u8>, v: &Vector3f) {
    push_float_bytes(bytes, v.x);
    push_float_bytes(bytes, v.y);
    push_float_bytes(bytes, v.z);
}

fn push_float_bytes(bytes: &mut Vec<u8>, value: Float) {
    bytes.extend_from_slice(&value.to_ne_bytes());
}

fn murmur_hash_64a(key: &[u8], seed: u64) -> u64 {
    let m = 0xc6a4a7935bd1e995u64;
    let r = 47u32;

    let len = key.len() as u64;
    let mut h = seed ^ len.wrapping_mul(m);

    let nblocks = key.len() / 8;
    for i in 0..nblocks {
        let start = i * 8;
        let mut k = u64::from_ne_bytes(key[start..start + 8].try_into().unwrap());
        k = k.wrapping_mul(m);
        k ^= k >> r;
        k = k.wrapping_mul(m);

        h ^= k;
        h = h.wrapping_mul(m);
    }

    let tail = &key[nblocks * 8..];
    let mut absorbed = false;
    if !tail.is_empty() {
        let mut hi = h;
        for (i, byte) in tail.iter().enumerate() {
            hi ^= (*byte as u64) << (i * 8);
        }
        h = hi.wrapping_mul(m);
        absorbed = true;
    }
    let _ = absorbed;

    h ^= h >> r;
    h = h.wrapping_mul(m);
    h ^= h >> r;
    h
}

pub fn sample_exponential(u: Float, a: Float) -> Float {
    -Float::ln(1.0 - u) / a
}

pub fn face_forward_z(v: Vector3f) -> Vector3f {
    if v.z < 0.0 {
        -v
    } else {
        v
    }
}

pub fn fr_dielectric(cos_theta_i: Float, _eta_i: Float, eta_t: Float) -> Float {
    super::specular::fr_dielectric(cos_theta_i, eta_t)
}

pub fn sqr(v: Float) -> Float {
    v * v
}
