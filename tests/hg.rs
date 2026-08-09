// Imported from hg.cpp

use pbrt_r4::prelude::*;

fn near_equal(a: Float, b: Float, e: Float) -> bool {
    (a - b).abs() < e
}

#[test]
fn henyey_greenstein_sampling_match() {
    let mut rng = RNG::new();
    let mut g = -0.75;
    while g <= 0.75 {
        let hg = HGPhaseFunction::new(g);
        for _ in 0..100 {
            let wo =
                uniform_sample_sphere(&Vector2f::new(rng.uniform_float(), rng.uniform_float()));
            let u = Vector2f::new(rng.uniform_float(), rng.uniform_float());
            let (p0, wi) = hg.sample_p(&wo, &u);
            assert!(near_equal(p0, hg.p(&wo, &wi), 1e-4));
        }
        g += 0.25;
    }
}

#[test]
fn henyey_greenstein_sampling_orientation_forward() {
    let mut rng = RNG::new();
    let hg = HGPhaseFunction::new(0.95);
    let wo = Vector3f::new(-1.0, 0.0, 0.0);

    let mut n_forward = 0;
    let mut n_backward = 0;
    for _ in 0..100 {
        let u = Vector2f::new(rng.uniform_float(), rng.uniform_float());
        let (_p0, wi) = hg.sample_p(&wo, &u);
        if wi.x > 0.0 {
            n_forward += 1;
        } else {
            n_backward += 1;
        }
        // With g = 0.95, almost all of the samples should have wi.x > 0.
        assert!(n_forward >= 10 * n_backward);
    }
}

#[test]
fn henyey_greenstein_normalized() {
    let mut rng = RNG::new();
    let mut g = -0.75;
    while g <= 0.75 {
        let hg = HGPhaseFunction::new(g);
        let wo = uniform_sample_sphere(&Vector2f::new(rng.uniform_float(), rng.uniform_float()));
        let mut sum = 0.0;
        let n_samples = 100000;
        for _ in 0..n_samples {
            let wi =
                uniform_sample_sphere(&Vector2f::new(rng.uniform_float(), rng.uniform_float()));
            sum += hg.p(&wo, &wi);
        }
        // Phase function should integrate to 1/4pi.
        assert!(near_equal(sum / n_samples as Float, 1.0 / (4.0 * PI), 1e-3));

        g += 0.25;
    }
}
