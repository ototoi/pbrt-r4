use pbrt_r4::util::base::Float;
use pbrt_r4::util::spectrum::config::{LAMBDA_MAX, LAMBDA_MIN, N_SPECTRUM_SAMPLES};
use pbrt_r4::util::spectrum::sampled::{SampledSpectrum, SampledWavelengths};

#[test]
fn sampled_spectrum_add_mul_average_work() {
    let a = SampledSpectrum::from([1.0, 2.0, 3.0, 4.0]);
    let b = SampledSpectrum::from([2.0, 3.0, 4.0, 5.0]);
    assert_eq!(a + b, SampledSpectrum::from([3.0, 5.0, 7.0, 9.0]));
    assert_eq!(a * 2.0, SampledSpectrum::from([2.0, 4.0, 6.0, 8.0]));
    assert!((a.average() - 2.5).abs() < 1e-6);
}

#[test]
fn sampled_wavelengths_visible_sampling_stays_in_range() {
    let lambda = SampledWavelengths::sample_visible(0.37);
    for i in 0..N_SPECTRUM_SAMPLES {
        assert!((LAMBDA_MIN as Float..=LAMBDA_MAX as Float).contains(&lambda[i]));
        assert!(lambda.pdf()[i] > 0.0);
    }
}

#[test]
fn terminate_secondary_matches_v4_behavior() {
    let mut lambda = SampledWavelengths::sample_uniform(0.25, 400.0, 700.0);
    let pdf0 = lambda.pdf()[0];
    lambda.terminate_secondary();
    assert!(lambda.secondary_terminated());
    assert!((lambda.pdf()[0] - pdf0 / N_SPECTRUM_SAMPLES as Float).abs() < 1e-6);
    assert_eq!(lambda.pdf()[1], 0.0);
    assert_eq!(lambda.pdf()[2], 0.0);
    assert_eq!(lambda.pdf()[3], 0.0);
}
