use pbrt_r4::gpu::flat::{RenderSettings, SamplerKind, SamplerRandomization};
use pbrt_r4::gpu::webgpu::sampler::build_sampler_data;
use pbrt_r4::util::lowdiscrepancy::primes::PRIMES;
use pbrt_r4::util::lowdiscrepancy::DigitPermutation;

fn halton_settings(samples_per_pixel: u32) -> RenderSettings {
    RenderSettings {
        sampler_kind: SamplerKind::Halton,
        randomization: SamplerRandomization::PermuteDigits,
        samples_per_pixel,
        x_samples: 1,
        y_samples: 1,
        jitter: false,
        max_depth: 5,
        seed: 17,
        light_sampler: "bvh".to_string(),
        disable_wavelength_jitter: false,
    }
}

#[test]
fn packed_permutations_match_cpu_digit_permutations() {
    let settings = halton_settings(16);
    let data = build_sampler_data(&settings, [320, 180]).unwrap();
    for dimension in 0..data.dimension_count() as usize {
        let header = dimension * 4;
        let expected = DigitPermutation::new(PRIMES[dimension] as u32, settings.seed);
        assert_eq!(data.table_words()[header], expected.base);
        assert_eq!(data.table_words()[header + 1], expected.n_digits);
        let offset = data.table_words()[header + 2] as usize;
        assert_eq!(
            &data.table_words()[offset..offset + expected.permutations.len()],
            expected
                .permutations
                .iter()
                .map(|&value| u32::from(value))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn invalid_halton_sample_ranges_are_rejected() {
    assert!(build_sampler_data(&halton_settings(0), [320, 180]).is_err());
    assert!(build_sampler_data(&halton_settings(u32::MAX), [320, 180]).is_err());
}

#[test]
fn sobol_supports_single_pixel_resolution() {
    let mut settings = halton_settings(16);
    settings.sampler_kind = SamplerKind::Sobol;
    settings.randomization = SamplerRandomization::FastOwen;
    assert!(build_sampler_data(&settings, [1, 1]).is_ok());
}

#[test]
fn pmj02bn_rejects_sample_counts_beyond_its_table() {
    let mut settings = halton_settings(65_537);
    settings.sampler_kind = SamplerKind::Pmj02Bn;
    settings.randomization = SamplerRandomization::None;
    assert!(build_sampler_data(&settings, [320, 180]).is_err());
}
