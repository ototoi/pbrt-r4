use super::beam_diffusion::BSSRDFTable;
use crate::base::bssrdf::{BSSRDFProbeSegment, BSSRDFSample, SubsurfaceInteraction};
use crate::base::bxdf::BxDF;
use crate::bsdf::BSDF;
use crate::bxdfs::NormalizedFresnelBxDF;
use crate::cpu::integrators::IntegratorBase;
use crate::interaction::SurfaceInteraction;
use crate::util::base::*;
use crate::util::geometry::coordinate_system;
use crate::util::interpolation::{catmull_rom_weights, sample_catmull_rom_2d};
use crate::util::spectrum::*;
use crate::util::vecmath::Frame;

use std::sync::Arc;

pub struct TabulatedBSSRDF {
    po: Point3f,
    wo: Vector3f,
    ns: Normal3f,
    eta: Float,
    table: Arc<BSSRDFTable>,
    // v4 stores these as SampledSpectrum (4 floats each, stack). r4 used
    // to store DenselySampledSpectrum (471-float Arc), which made the
    // per-shade BSSRDF construction allocate ~10 KB and forced every
    // sr/pdf loop to iterate 471 channels instead of 4. Matching v4
    // verbatim eliminates the alloc and shrinks the inner loop ~100x.
    sigma_t: SampledSpectrum,
    rho: SampledSpectrum,
}

impl TabulatedBSSRDF {
    pub fn new(
        po: Point3f,
        ns: Normal3f,
        wo: Vector3f,
        eta: Float,
        sigma_a: SampledSpectrum,
        sigma_s: SampledSpectrum,
        table: Arc<BSSRDFTable>,
    ) -> Self {
        let sigma_t = sigma_a + sigma_s;
        let mut rho_values = [0.0; SampledSpectrum::N_SAMPLES];
        for i in 0..SampledSpectrum::N_SAMPLES {
            rho_values[i] = if sigma_t[i] > 0.0 {
                sigma_s[i] / sigma_t[i]
            } else {
                0.0
            };
        }
        let rho = SampledSpectrum::from(rho_values);

        Self {
            po,
            wo,
            ns,
            eta,
            table,
            sigma_t,
            rho,
        }
    }

    fn sr(&self, r: Float) -> SampledSpectrum {
        let mut sr_values = [0.0; SampledSpectrum::N_SAMPLES];
        for ch in 0..SampledSpectrum::N_SAMPLES {
            sr_values[ch] = Float::max(0.0, self.sr_core(ch, r).0);
        }
        SampledSpectrum::from(sr_values)
    }

    fn pdf_sr(&self, ch: usize, r: Float) -> Float {
        let (sr, rho_eff) = self.sr_core(ch, r);
        Float::max(0.0, sr / rho_eff)
    }

    fn sr_core(&self, ch: usize, r: Float) -> (Float, Float) {
        let r_optical = r * self.sigma_t[ch];
        if let Some((rho_offset, rho_weights)) =
            catmull_rom_weights(&self.table.rho_samples, self.rho[ch])
        {
            if let Some((radius_offset, radius_weights)) =
                catmull_rom_weights(&self.table.radius_samples, r_optical)
            {
                let mut sr = 0.0;
                let mut rho_eff = 0.0;
                for i in 0..4 {
                    if rho_weights[i] == 0.0 {
                        continue;
                    }
                    let rho_index = (rho_offset + i as i32) as usize;
                    rho_eff += self.table.rho_eff[rho_index] * rho_weights[i];
                    for j in 0..4 {
                        if radius_weights[j] == 0.0 {
                            continue;
                        }
                        let radius_index = (radius_offset + j as i32) as usize;
                        sr += self.table.eval_profile(rho_index, radius_index)
                            * rho_weights[i]
                            * radius_weights[j];
                    }
                }
                if r_optical != 0.0 {
                    sr /= 2.0 * PI * r_optical;
                }
                let sigma2 = self.sigma_t[ch] * self.sigma_t[ch];
                return (sr * sigma2, rho_eff);
            }
        }
        (0.0, 1.0)
    }
}

impl TabulatedBSSRDF {
    /// Per-wavelength `Sp` (v4 `Sp(pi)` returns `SampledSpectrum`).
    /// Mirrors `sr` above but the result is the `SampledSpectrum`
    /// shape used by the rescaled-density MIS book-keeping in
    /// `VolPathIntegrator`.
    fn sp_packet(&self, pi: Point3f, _lambda: &SampledWavelengths) -> SampledSpectrum {
        // `sr` already returns `SampledSpectrum`; the previous
        // `.sample(lambda)` round-trip via Spectrum was wasteful (and
        // wrong with the new SampledSpectrum-based BSSRDF — there's no
        // dense form to project from).
        let r = Vector3f::distance(&self.po, &pi);
        self.sr(r)
    }

    /// Per-wavelength `PDF_Sp` matching pbrt-v4 (`bssrdf.h:235-254`).
    /// The probe-axis projection is the same as the scalar `pdf_sp`,
    /// but the per-channel integrand keeps `pdf_sr(ch, r)` per channel
    /// instead of summing across channels.
    fn pdf_sp_packet(
        &self,
        pi: Point3f,
        ni: Normal3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        // Project (pi - po) and ni into the local shading-normal frame.
        let d = pi - self.po;
        let frame = Frame::from_z(Vector3f::from(self.ns));
        let d_local = frame.to_local(d);
        let n_local = frame.to_local(Vector3f::from(ni));

        let r_proj = [
            Float::sqrt(d_local.y * d_local.y + d_local.z * d_local.z),
            Float::sqrt(d_local.z * d_local.z + d_local.x * d_local.x),
            Float::sqrt(d_local.x * d_local.x + d_local.y * d_local.y),
        ];
        let axis_prob = [0.25, 0.25, 0.5];

        // For each (axis, channel) accumulate per-channel pdf. With
        // sigma_t/rho now `SampledSpectrum`, "channel" iterates 4 not
        // 471 — matches v4's NSpectrumSamples loop.
        let _ = lambda;
        let mut pdf_values = [0.0; SampledSpectrum::N_SAMPLES];
        for axis in 0..3 {
            let n_abs = n_local[axis].abs();
            if n_abs == 0.0 {
                continue;
            }
            for ch in 0..SampledSpectrum::N_SAMPLES {
                pdf_values[ch] += self.pdf_sr(ch, r_proj[axis]) * n_abs * axis_prob[axis];
            }
        }
        SampledSpectrum::from(pdf_values).clamp_zero()
    }

    /// pbrt-v4 `TabulatedBSSRDF::SampleSp(u1, u2)` — produce just the
    /// probe-segment endpoints; the caller walks the scene between
    /// them and feeds the same-material intersections into a
    /// `WeightedReservoirSampler`. Takes `lambda` because the
    /// distance distribution depends on `sigma_t` / `rho` at the
    /// **hero wavelength** of the current `SampledWavelengths`,
    /// matching v4. Using a fixed dense-spectrum index biased the
    /// result by ~4% on the `head` BSSRDF scene.
    pub fn sample_sp_v4(
        &self,
        u1: Float,
        u2: &Point2f,
        lambda: &SampledWavelengths,
    ) -> Option<BSSRDFProbeSegment> {
        // sigma_t/rho are now SampledSpectrum (4 floats), already
        // evaluated at the current pixel's `lambda` packet by the
        // caller; index 0 is the hero wavelength.
        let _ = lambda;
        let sigma_t0 = self.sigma_t[0];
        let rho0 = self.rho[0];
        if sigma_t0 == 0.0 {
            return None;
        }

        let frame = if u1 < 0.25 {
            Frame::from_x(Vector3f::from(self.ns))
        } else if u1 < 0.5 {
            Frame::from_y(Vector3f::from(self.ns))
        } else {
            Frame::from_z(Vector3f::from(self.ns))
        };

        let r = sample_catmull_rom_2d_distance(
            &self.table.rho_samples,
            &self.table.radius_samples,
            &self.table.profile,
            &self.table.profile_cdf,
            rho0,
            u2[0],
        )?;
        let r = r / sigma_t0;

        let r_max = sample_catmull_rom_2d_distance(
            &self.table.rho_samples,
            &self.table.radius_samples,
            &self.table.profile,
            &self.table.profile_cdf,
            rho0,
            0.999,
        )?;
        let r_max = r_max / sigma_t0;
        if r >= r_max {
            return None;
        }
        let l = 2.0 * Float::sqrt(r_max * r_max - r * r);
        if l <= 1e-6 {
            return None;
        }
        let phi = 2.0 * PI * u2[1];

        let p_start = self.po + r * (frame.x * Float::cos(phi) + frame.y * Float::sin(phi))
            - 0.5 * l * frame.z;
        let p_target = p_start + l * frame.z;
        Some(BSSRDFProbeSegment::new(p_start, p_target))
    }
}

fn sample_catmull_rom_2d_distance(
    rho_samples: &[Float],
    radius_samples: &[Float],
    profile: &[Float],
    profile_cdf: &[Float],
    rho_value: Float,
    u: Float,
) -> Option<Float> {
    sample_catmull_rom_2d(
        rho_samples,
        radius_samples,
        profile,
        profile_cdf,
        rho_value,
        u,
    )
    .map(|(v, _, _)| v)
}

impl TabulatedBSSRDF {
    pub fn sample_s(
        &self,
        integrator: &IntegratorBase,
        u1: Float,
        u2: &Point2f,
    ) -> Option<(SampledSpectrum, SurfaceInteraction, Float)> {
        let lambda = SampledWavelengths::sample_visible(0.5);
        let probe = self.sample_sp_v4(u1, u2, &lambda)?;

        let mut base = SurfaceInteraction {
            p: probe.p0,
            ..Default::default()
        };
        let mut chain = Vec::new();
        loop {
            let distance = Vector3f::distance(&base.p, &probe.p1);
            if distance <= MACHINE_EPSILON {
                break;
            }

            let ray = base.spawn_ray_to_point(&probe.p1);
            if let Some(si) = integrator.intersect(&ray, 1.0 - SHADOW_EPSILON) {
                base = si.intr.clone();
                chain.push(si.intr);
            } else {
                break;
            }
        }

        if chain.is_empty() {
            return None;
        }

        let selected = usize::clamp((u1 * chain.len() as Float) as usize, 0, chain.len() - 1);
        let mut si = chain[selected].clone();
        let sp = self.sp_packet(si.p, &lambda);
        let pdf = self.pdf_sp_packet(si.p, si.n, &lambda)[0] / chain.len() as Float;
        if !sp.is_black() {
            let (ns, dpdu) = shading_frame(&si);
            let bxdf = BxDF::NormalizedFresnel(Box::new(NormalizedFresnelBxDF::new(self.eta)));
            si.bsdf = Some(BSDF::new(ns, dpdu, bxdf));
            si.wo = Vector3f::from(ns);
        }
        Some((sp, si, pdf))
    }

    pub fn sample_sp(
        &self,
        u1: Float,
        u2: &Point2f,
        lambda: &SampledWavelengths,
    ) -> Option<BSSRDFProbeSegment> {
        Self::sample_sp_v4(self, u1, u2, lambda)
    }

    pub fn probe_intersection_to_sample(
        &self,
        ssi: &SubsurfaceInteraction,
        lambda: &SampledWavelengths,
    ) -> Option<BSSRDFSample> {
        let sp = self.sp_packet(ssi.p, lambda);
        let pdf = self.pdf_sp_packet(ssi.p, ssi.n, lambda);
        if sp.is_black() || pdf[0] <= 0.0 {
            return None;
        }
        let wo = Vector3f::from(ssi.ns);
        let bxdf = BxDF::NormalizedFresnel(Box::new(NormalizedFresnelBxDF::new(self.eta)));
        let bsdf = BSDF::new(ssi.ns, ssi.dpdus, bxdf);
        Some(BSSRDFSample {
            sp,
            pdf,
            sw: bsdf,
            wo,
        })
    }
}

impl std::fmt::Debug for TabulatedBSSRDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TabulatedBSSRDF")
            .field("po", &self.po)
            .field("wo", &self.wo)
            .field("ns", &self.ns)
            .field("eta", &self.eta)
            .finish()
    }
}

fn shading_frame(si: &SurfaceInteraction) -> (Normal3f, Vector3f) {
    let ns = if si.shading.n.length_squared() > 0.0 {
        si.shading.n
    } else if si.n.length_squared() > 0.0 {
        si.n
    } else {
        Normal3f::new(0.0, 0.0, 1.0)
    };

    let tangent = [
        si.shading.dpdu,
        si.dpdu,
        Vector3f::new(1.0, 0.0, 0.0),
        coordinate_system(&Vector3f::from(ns)).0,
    ]
    .into_iter()
    .find(|v| {
        v.length_squared() > 0.0 && Vector3f::cross(&Vector3f::from(ns), v).length_squared() > 0.0
    })
    .unwrap_or_else(|| coordinate_system(&Vector3f::from(ns)).0)
    .normalize();

    (ns, tangent)
}
