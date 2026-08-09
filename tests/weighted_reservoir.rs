use pbrt_r4::util::sampling::weighted_reservoir::WeightedReservoirSampler;

#[test]
fn empty_reservoir_has_no_sample() {
    let r: WeightedReservoirSampler<i32> = WeightedReservoirSampler::new();
    assert!(!r.has_sample());
    assert_eq!(r.sample_probability(), 0.0);
}

#[test]
fn single_add_always_keeps() {
    let mut r = WeightedReservoirSampler::with_seed(1);
    r.add(42i32, 1.0);
    assert!(r.has_sample());
    assert_eq!(r.sample(), Some(&42));
    assert_eq!(r.sample_probability(), 1.0);
}

#[test]
fn many_equal_weights_are_uniformly_distributed() {
    let mut r = WeightedReservoirSampler::with_seed(1234);
    for i in 0..1000 {
        r.add(i, 1.0);
    }
    assert!(r.has_sample());
    let idx = *r.sample().unwrap();
    assert!((0..1000).contains(&idx));
    assert!((r.sample_probability() - 1.0 / 1000.0).abs() < 1e-6);
}

#[test]
fn zero_weight_is_ignored() {
    let mut r = WeightedReservoirSampler::with_seed(7);
    r.add(1, 0.0);
    assert!(!r.has_sample());
}
