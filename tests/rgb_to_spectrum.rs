use pbrt_r4::util::spectrum::rgb_to_spectrum::{d65_sample, SRGB};
use pbrt_r4::util::spectrum::sampled::{SampledSpectrum, SampledWavelengths};

#[test]
fn rgb_color_space_exposes_illuminant() {
    let lambda = SampledWavelengths::sample_visible(0.37);
    let sampled = SRGB.illuminant.sample(&lambda);
    for i in 0..SampledSpectrum::N_SAMPLES {
        assert!((sampled[i] - d65_sample(lambda[i])).abs() < 1e-6);
    }

    let dense = SRGB.illuminant.to_dense();
    assert!((dense.sample_at(560.0) - d65_sample(560.0)).abs() < 1e-6);
}

#[test]
fn color_space_illuminant_drives_rgb_illuminant_sampling() {
    let lambda = SampledWavelengths::sample_visible(0.42);
    let rgb = [1.0, 1.0, 1.0];
    let sampled = SRGB.illuminant_to_sampled_spectrum(rgb, &lambda);

    for i in 0..SampledSpectrum::N_SAMPLES {
        assert!((sampled[i] - SRGB.illuminant.sample_at(lambda[i])).abs() < 1e-6);
    }
}
