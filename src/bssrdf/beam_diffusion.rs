use crate::media::phase_hg;
use crate::util::base::*;
use crate::util::interpolation::{integrate_catmull_rom, invert_catmull_rom};
use crate::util::spectrum::*;

#[derive(Debug)]
pub struct BSSRDFTable {
    pub rho_samples: Vec<Float>,
    pub radius_samples: Vec<Float>,
    pub profile: Vec<Float>,
    pub rho_eff: Vec<Float>,
    pub profile_cdf: Vec<Float>,
}

impl BSSRDFTable {
    pub fn new(n_rho_samples: usize, n_radius_samples: usize) -> Self {
        Self {
            rho_samples: vec![0.0; n_rho_samples],
            radius_samples: vec![0.0; n_radius_samples],
            profile: vec![0.0; n_rho_samples * n_radius_samples],
            rho_eff: vec![0.0; n_rho_samples],
            profile_cdf: vec![0.0; n_rho_samples * n_radius_samples],
        }
    }

    pub fn eval_profile(&self, rho_index: usize, radius_index: usize) -> Float {
        self.profile[rho_index * self.radius_samples.len() + radius_index]
    }
}

const N_SAMPLES: usize = 100;

pub fn fresnel_moment1(eta: Float) -> Float {
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let eta4 = eta3 * eta;
    let eta5 = eta4 * eta;
    if eta < 1.0 {
        0.45966 - 1.73965 * eta + 3.37668 * eta2 - 3.904945 * eta3 + 2.49277 * eta4 - 0.68441 * eta5
    } else {
        -4.61686 + 11.1136 * eta - 10.4646 * eta2 + 5.11455 * eta3 - 1.27198 * eta4 + 0.12746 * eta5
    }
}

fn fresnel_moment2(eta: Float) -> Float {
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let eta4 = eta3 * eta;
    let eta5 = eta4 * eta;
    if eta < 1.0 {
        0.27614 - 0.87350 * eta + 1.12077 * eta2 - 0.65095 * eta3 + 0.07883 * eta4 + 0.04860 * eta5
    } else {
        let r_eta = 1.0 / eta;
        let r_eta2 = r_eta * r_eta;
        let r_eta3 = r_eta2 * r_eta;
        -547.033 + 45.3087 * r_eta3 - 218.725 * r_eta2 + 458.843 * r_eta + 404.557 * eta
            - 189.519 * eta2
            + 54.9327 * eta3
            - 9.00603 * eta4
            + 0.63942 * eta5
    }
}

pub fn subsurface_from_diffuse(
    table: &BSSRDFTable,
    rho_eff: &Spectrum,
    mfp: &Spectrum,
) -> (Spectrum, Spectrum) {
    let rho_eff = rho_eff.to_dense();
    let mfp = mfp.to_dense();
    let mut sigma_a = DenseSampledSpectrum::zero();
    let mut sigma_s = DenseSampledSpectrum::zero();
    for c in 0..DenseSampledSpectrum::N_SAMPLES {
        let rho = invert_catmull_rom(&table.rho_samples, &table.rho_eff, rho_eff[c]);
        let mfp_c = mfp[c].max(1e-6);
        sigma_s[c] = rho / mfp_c;
        sigma_a[c] = (1.0 - rho) / mfp_c;
    }
    (Spectrum::from(&sigma_a), Spectrum::from(&sigma_s))
}

/// pbrt-v4 `SubsurfaceFromDiffuse(table, rho_eff, mfp, &sigma_a,
/// &sigma_s)` operating directly on `SampledSpectrum` packets (4 floats
/// each, stack). Matches the v4 path in `materials.h:762`.
pub fn subsurface_from_diffuse_sampled(
    table: &BSSRDFTable,
    rho_eff: &SampledSpectrum,
    mfp: &SampledSpectrum,
) -> (SampledSpectrum, SampledSpectrum) {
    let mut sigma_a = [0.0; SampledSpectrum::N_SAMPLES];
    let mut sigma_s = [0.0; SampledSpectrum::N_SAMPLES];
    for c in 0..SampledSpectrum::N_SAMPLES {
        let rho = invert_catmull_rom(&table.rho_samples, &table.rho_eff, rho_eff[c]);
        let mfp_c = mfp[c].max(1e-6);
        sigma_s[c] = rho / mfp_c;
        sigma_a[c] = (1.0 - rho) / mfp_c;
    }
    (
        SampledSpectrum::from(sigma_a),
        SampledSpectrum::from(sigma_s),
    )
}

pub fn compute_beam_diffusion_bssrdf(g: Float, eta: Float, table: &mut BSSRDFTable) {
    let n_radius_samples = table.radius_samples.len();
    table.radius_samples[0] = 0.0;
    table.radius_samples[1] = 2.5e-3;
    for i in 2..n_radius_samples {
        table.radius_samples[i] = table.radius_samples[i - 1] * 1.2;
    }

    let n_rho_samples = table.rho_samples.len();
    for i in 0..n_rho_samples {
        let a = 1.0 - Float::exp(-8.0 * (i as Float) / (n_rho_samples - 1) as Float);
        let b = 1.0 - Float::exp(-8.0);
        table.rho_samples[i] = a / b;
    }

    for i in 0..n_rho_samples {
        for j in 0..n_radius_samples {
            let rho = table.rho_samples[i];
            let r = table.radius_samples[j];
            let ss = beam_diffusion_ss(rho, 1.0 - rho, g, eta, r);
            let ms = beam_diffusion_ms(rho, 1.0 - rho, g, eta, r);
            table.profile[i * n_radius_samples + j] = 2.0 * PI * r * (ss + ms);
        }

        let start = i * n_radius_samples;
        let end = start + n_radius_samples;
        let cdf = integrate_catmull_rom(&table.radius_samples, &table.profile[start..end]);
        table.rho_eff[i] = cdf[n_radius_samples - 1];
        table.profile_cdf[start..end].copy_from_slice(&cdf);
    }
}

fn beam_diffusion_ms(sigma_s: Float, sigma_a: Float, g: Float, eta: Float, r: Float) -> Float {
    let mut ed = 0.0;
    let sigmap_s = sigma_s * (1.0 - g);
    let sigmap_t = sigma_a + sigmap_s;
    let rhop = sigmap_s / sigmap_t;

    let d_g = (2.0 * sigma_a + sigmap_s) / (3.0 * sigmap_t * sigmap_t);
    let sigma_tr = Float::sqrt(sigma_a / d_g);

    let fm1 = fresnel_moment1(eta);
    let fm2 = fresnel_moment2(eta);
    let ze = -2.0 * d_g * (1.0 + 3.0 * fm2) / (1.0 - 2.0 * fm1);

    let c_phi = 0.25 * (1.0 - 2.0 * fm1);
    let c_e = 0.5 * (1.0 - 3.0 * fm2);
    for i in 0..N_SAMPLES {
        let zr = -Float::ln(1.0 - (i as Float + 0.5) / (N_SAMPLES as Float)) / sigmap_t;
        let zv = -zr + 2.0 * ze;
        let dr = Float::sqrt(r * r + zr * zr);
        let dv = Float::sqrt(r * r + zv * zv);

        let phi_d =
            INV_4_PI / d_g * (Float::exp(-sigma_tr * dr) / dr - Float::exp(-sigma_tr * dv) / dv);
        let edn = INV_4_PI
            * (zr * (1.0 + sigma_tr * dr) * Float::exp(-sigma_tr * dr) / (dr * dr * dr)
                - zv * (1.0 + sigma_tr * dv) * Float::exp(-sigma_tr * dv) / (dv * dv * dv));

        let e = phi_d * c_phi + edn * c_e;
        let kappa = 1.0 - Float::exp(-2.0 * sigmap_t * (dr + zr));
        ed += kappa * rhop * rhop * e;
    }

    ed / N_SAMPLES as Float
}

fn beam_diffusion_ss(sigma_s: Float, sigma_a: Float, g: Float, eta: Float, r: Float) -> Float {
    let sigma_t = sigma_a + sigma_s;
    let rho = sigma_s / sigma_t;
    let t_crit = r * Float::sqrt(eta * eta - 1.0);
    let mut ess = 0.0;

    for i in 0..N_SAMPLES {
        let ti = t_crit - Float::ln(1.0 - (i as Float + 0.5) / (N_SAMPLES as Float)) / sigma_t;
        let d = Float::sqrt(r * r + ti * ti);
        let cos_theta_o = ti / d;

        ess += rho * Float::exp(-sigma_t * (d + t_crit)) / (d * d)
            * phase_hg(cos_theta_o, g)
            * (1.0 - fr_dielectric(-cos_theta_o, 1.0, eta))
            * Float::abs(cos_theta_o);
    }
    ess / N_SAMPLES as Float
}

fn fr_dielectric(mut cos_theta_i: Float, mut eta_i: Float, mut eta_t: Float) -> Float {
    cos_theta_i = Float::clamp(cos_theta_i, -1.0, 1.0);
    let entering = cos_theta_i > 0.0;
    if !entering {
        std::mem::swap(&mut eta_i, &mut eta_t);
        cos_theta_i = Float::abs(cos_theta_i);
    }

    let sin_theta_i = Float::sqrt(Float::max(0.0, 1.0 - cos_theta_i * cos_theta_i));
    let sin_theta_t = eta_i / eta_t * sin_theta_i;
    if sin_theta_t >= 1.0 {
        return 1.0;
    }

    let cos_theta_t = Float::sqrt(Float::max(0.0, 1.0 - sin_theta_t * sin_theta_t));
    let r_parl = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t))
        / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
    let r_perp = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t))
        / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));
    0.5 * (r_parl * r_parl + r_perp * r_perp)
}
