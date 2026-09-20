use pbrt_r4::gpu::flat::{HaltonRandomization, RenderSettings, SamplerKind};
use pbrt_r4::gpu::webgpu::sampler::build_sampler_data;
use pbrt_r4::util::lowdiscrepancy::primes::PRIMES;
use pbrt_r4::util::lowdiscrepancy::DigitPermutation;

fn halton_settings(samples_per_pixel: u32) -> RenderSettings {
    RenderSettings {
        sampler_kind: SamplerKind::Halton,
        halton_randomization: HaltonRandomization::PermuteDigits,
        samples_per_pixel,
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
    for dimension in 0..data.uniform.dimension_count as usize {
        let header = dimension * 4;
        let expected = DigitPermutation::new(PRIMES[dimension] as u32, settings.seed);
        assert_eq!(data.words[header], expected.base);
        assert_eq!(data.words[header + 1], expected.n_digits);
        let offset = data.words[header + 2] as usize;
        assert_eq!(
            &data.words[offset..offset + expected.permutations.len()],
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
