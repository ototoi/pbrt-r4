use pbrt_r4::util::rng::{mix_bits, RNG};

#[test]
fn test_001() {
    let a: u32 = 0b10100000;
    let b: u32 = 0b01011111;
    assert_eq!((!a) & 0b11111111, b);
}

#[test]
fn test_002() {
    let mut rng = RNG::new();
    let a: f32 = rng.uniform_float32();
    let astate = rng.state;
    let b: f32 = rng.uniform_float32();
    let bstate = rng.state;
    assert_ne!(a, b);
    assert_ne!(astate, bstate);
}

#[test]
fn integer_and_double_uniforms_follow_the_pcg_sequence() {
    let mut combined = RNG::new();
    let hi = u64::from(combined.uniform_uint32());
    let lo = u64::from(combined.uniform_uint32());
    let expected = (hi << 32) | lo;

    let mut rng = RNG::new();
    assert_eq!(rng.uniform_uint64(), expected);
    let sample = rng.uniform_float64();
    assert!((0.0..1.0).contains(&sample));
}

#[test]
fn distance_matches_advance() {
    let start = RNG::new_sequence(17);
    let mut end = start.clone();
    end.advance(37);
    assert_eq!(end.distance(&start), Ok(37));
    assert!(start.distance(&RNG::new_sequence(18)).is_err());
}

#[test]
fn one_argument_sequence_uses_v4_mix_bits_seed() {
    let mut sequence = RNG::new_sequence(17);
    let mut explicit = RNG::new();
    explicit.set_sequence_with_seed(17, mix_bits(17));
    for _ in 0..4 {
        assert_eq!(sequence.uniform_uint32(), explicit.uniform_uint32());
    }
}
