// Imported from hair.cpp
/*
use pbrt_r4::prelude::*;
use pbrt_r4::materials::hair::*;

#[test]
fn hair_white_furnace() {
    let mut rng = RNG::new();
    let wo = uniform_sample_sphere(&Vector2f::new(rng.uniform_float(), rng.uniform_float()));
    let mut beta_m = 0.1;
    while beta_m < 1.0 {
        let mut beta_n = 0.1;
        while beta_n < 1.0 {
            // Estimate reflected uniform incident radiance from hair
            let mut sum = Spectrum::zero();
            let count = 300000;
            for _ in 0..count {
                let h = -1.0 + 2.0 * rng.uniform_float();
                let sigma_a = Spectrum::zero();
                let hair = HairBSDF::new(h, 1.55, sigma_a, beta_m, beta_n, 0.0);
                let wi =
                    uniform_sample_sphere(&Vector2f::new(rng.uniform_float(), rng.uniform_float()));
                sum += hair.f(&wo, &wi) * abs_cos_theta(&wi);
            }
            let avg = sum.y() / (count as Float * uniform_sphere_pdf());
            assert!(avg >= 0.95 && avg <= 1.05, "avg = {}", avg);
            beta_n += 0.2;
        }
        beta_m += 0.2;
    }
}

#[test]
fn hair_white_furnace_sampled() {
    let mut rng = RNG::new();
    let wo = uniform_sample_sphere(&Vector2f::new(rng.uniform_float(), rng.uniform_float()));
    let mut beta_m = 0.1;
    while beta_m < 1.0 {
        let mut beta_n = 0.1;
        while beta_n < 1.0 {
            let mut sum = Spectrum::zero();
            let count = 300000;
            for _ in 0..count {
                let h = -1.0 + 2.0 * rng.uniform_float();
                let sigma_a = Spectrum::zero();
                let hair = HairBSDF::new(h, 1.55, sigma_a, beta_m, beta_n, 0.0);
                let u = Point2f::new(rng.uniform_float(), rng.uniform_float());
                if let Some((f, wi, pdf, _t)) = hair.sample_f(&wo, &u) {
                    sum += f * abs_cos_theta(&wi) * (1.0 / pdf);
                }
            }
            let avg = sum.y() / count as Float;
            assert!(avg >= 0.99 && avg <= 1.01, "avg = {}", avg);
            beta_n += 0.2;
        }
        beta_m += 0.2;
    }
}

#[test]
fn hair_sampling_weights() {
    let mut rng = RNG::new();
    let mut beta_m = 0.1;
    while beta_m < 1.0 {
        let mut beta_n = 0.4;
        while beta_n < 1.0 {
            let count = 10000;
            for _ in 0..count {
                let h = -1.0 + 2.0 * rng.uniform_float();
                let sigma_a = Spectrum::zero();
                let hair = HairBSDF::new(h, 1.55, sigma_a, beta_m, beta_n, 0.0);
                let wo =
                    uniform_sample_sphere(&Vector2f::new(rng.uniform_float(), rng.uniform_float()));
                let u = Point2f::new(rng.uniform_float(), rng.uniform_float());
                if let Some((f, wi, pdf, _t)) = hair.sample_f(&wo, &u) {
                    // Verify that hair BSDF sample weight is close to 1 for
                    // _wi_
                    let weight = f.y() * abs_cos_theta(&wi) * (1.0 / pdf);
                    assert!(weight >= 0.99, "weight = {}", weight);
                    assert!(weight <= 1.01, "weight = {}", weight);
                }
            }
            beta_n += 0.2;
        }
        beta_m += 0.2;
    }
}

#[test]
fn hair_sampling_consistency() {
    let li = |w: &Vector3f| w.z * w.z;
    let mut rng = RNG::new();
    let mut beta_m = 0.1;
    while beta_m < 1.0 {
        let mut beta_n = 0.4;
        while beta_n < 1.0 {
            let count = 64 * 1024;
            let sigma_a = Spectrum::from(0.25);
            let wo =
                uniform_sample_sphere(&Vector2f::new(rng.uniform_float(), rng.uniform_float()));
            let mut f_importance = Spectrum::zero();
            let mut f_uniform = Spectrum::zero();
            for _ in 0..count {
                // Compute estimates of scattered radiance for hair sampling
                // test
                let h = -1.0 + 2.0 * rng.uniform_float();
                let hair = HairBSDF::new(h, 1.55, sigma_a, beta_m, beta_n, 0.0);
                let u = Point2f::new(rng.uniform_float(), rng.uniform_float());
                if let Some((f, wi, pdf, _t)) = hair.sample_f(&wo, &u) {
                    f_importance +=
                        f * li(&wi) * abs_cos_theta(&wi) * (1.0 / (pdf * count as Float));
                }
                let wi = uniform_sample_sphere(&u);
                f_uniform += hair.f(&wo, &wi)
                    * li(&wi)
                    * abs_cos_theta(&wi)
                    * (1.0 / (uniform_sphere_pdf() * count as Float));
            }
            // Verify consistency of estimated hair reflected radiance values
            let err = Float::abs(f_importance.y() - f_uniform.y()) / f_uniform.y();
            assert!(err < 0.05, "err = {}", err);
            beta_n += 0.2;
        }
        beta_m += 0.2;
    }
}
*/
