#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::{build_light_bounds, Bounds3, LightBoundInput, LightBounds};
use pbrt_r4::gpu::webgpu::light_sampler::{
    build_power_alias_table, pack_exhaustive_bounds, pack_power_weights, EXHAUSTIVE_BOUNDS_WORDS,
};

#[test]
fn power_alias_table_preserves_distribution_and_zero_weight_fallback() {
    let bins = build_power_alias_table(&[1.0, 3.0]).unwrap();
    assert_eq!(
        bins.iter().map(|bin| bin.p).collect::<Vec<_>>(),
        [0.25, 0.75]
    );
    assert_eq!(bins[0].q, 0.5);
    assert_eq!(bins[0].alias, 1);
    assert_eq!(bins[1].q, 1.0);

    let uniform = build_power_alias_table(&[0.0, 0.0]).unwrap();
    assert_eq!(
        uniform.iter().map(|bin| bin.p).collect::<Vec<_>>(),
        [0.5, 0.5]
    );
    assert!(build_power_alias_table(&[1.0, f32::NAN]).is_err());

    let packed = pack_power_weights(&[1.0, 3.0]).unwrap();
    assert_eq!(packed.len(), 7);
    assert_eq!(packed[0], 2);
    assert_eq!(f32::from_bits(packed[1]), 0.5);
    assert_eq!(f32::from_bits(packed[2]), 0.25);
    assert_eq!(packed[3], 1);
}

#[test]
fn exhaustive_bounds_pack_in_global_handle_order() {
    let bounds = [
        LightBounds {
            bounds: Bounds3::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).unwrap(),
            direction: [0.0, 0.0, 1.0],
            phi: 7.0,
            cos_theta_o: 0.8,
            cos_theta_e: 0.1,
            two_sided: true,
        },
        LightBounds {
            bounds: Bounds3::new([-1.0, -2.0, -3.0], [0.0, 0.0, 0.0]).unwrap(),
            direction: [0.0, 1.0, 0.0],
            phi: 8.0,
            cos_theta_o: 0.7,
            cos_theta_e: 0.2,
            two_sided: false,
        },
    ];
    let words = pack_exhaustive_bounds(&bounds).unwrap();
    assert_eq!(words.len(), 1 + 2 * EXHAUSTIVE_BOUNDS_WORDS);
    assert_eq!(words[0], 2);
    assert_eq!(words[1], 0);
    assert_eq!(words[1 + EXHAUSTIVE_BOUNDS_WORDS], 1);
    assert_eq!(f32::from_bits(words[11]), 7.0);
    assert_eq!(words[14], 1);
}

#[test]
fn spot_bound_keeps_v4_direction_and_emission_cone() {
    let bounds = build_light_bounds(&[LightBoundInput::Spot {
        handle: 0,
        world_position: [1.0, 2.0, 3.0],
        direction: [0.0, 2.0, 0.0],
        intensity_max: 2.0,
        scale: 3.0,
        cos_theta_o: 0.9,
        cos_falloff_end: 0.8,
    }])
    .unwrap()[0];
    assert_eq!(bounds.bounds.min, [1.0, 2.0, 3.0]);
    assert_eq!(bounds.direction, [0.0, 1.0, 0.0]);
    assert_eq!(bounds.phi, 4.0 * std::f32::consts::PI * 6.0);
    assert_eq!(bounds.cos_theta_o, 0.9);
    assert_eq!(bounds.cos_theta_e, (0.8f32.acos() - 0.9f32.acos()).cos());
}

#[test]
fn power_alias_normalization_uses_v4_double_accumulation() {
    let weights = [16777216.0, 1.0, 1.0];
    let bins = build_power_alias_table(&weights).unwrap();
    let expected_sum = 16777218.0f32;
    for (bin, weight) in bins.iter().zip(weights) {
        assert_eq!(bin.p, weight / expected_sum);
    }
    let mut outcome_probabilities = vec![0.0f64; bins.len()];
    for (index, bin) in bins.iter().enumerate() {
        outcome_probabilities[index] += f64::from(bin.q) / bins.len() as f64;
        if bin.q < 1.0 {
            outcome_probabilities[bin.alias as usize] +=
                (1.0 - f64::from(bin.q)) / bins.len() as f64;
        }
    }
    for (actual, bin) in outcome_probabilities.iter().zip(&bins) {
        assert!((actual - f64::from(bin.p)).abs() < 2e-8);
    }
}

#[test]
fn power_alias_sampling_never_selects_zero_weight_bins() {
    let bins = build_power_alias_table(&[0.0, 2.0, 0.0]).unwrap();
    for step in 0..4096 {
        let scaled = (step as f32 + 0.5) / 4096.0 * bins.len() as f32;
        let index = scaled as usize;
        let bin = bins[index];
        let selected = if scaled - (index as f32) < bin.q {
            index
        } else {
            bin.alias as usize
        };
        assert_eq!(selected, 1);
        assert_eq!(bins[selected].p, 1.0);
    }
}
