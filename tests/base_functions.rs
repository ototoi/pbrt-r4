use pbrt_r4::util::base::{log2int, log4_int_64, log4int, quadratic, round_up_pow2};

#[test]
fn quadratic_handles_linear_and_repeated_roots() {
    assert_eq!(quadratic(0.0, 2.0, -4.0), Some((2.0, 2.0)));
    let roots = quadratic(1.0, -2.0, 1.0).unwrap();
    assert!((roots.0 - 1.0).abs() < 1e-6);
    assert!((roots.1 - 1.0).abs() < 1e-6);
}

#[test]
fn round_up_and_log_helpers_match_expected_values() {
    assert_eq!(round_up_pow2(1), 1);
    assert_eq!(round_up_pow2(2), 2);
    assert_eq!(round_up_pow2(3), 4);
    assert_eq!(round_up_pow2(4), 4);
    assert_eq!(round_up_pow2(5), 8);

    assert_eq!(log2int(2), 1);
    assert_eq!(log2int(3), 1);
    assert_eq!(log2int(4), 2);
    assert_eq!(log4int(4), 1);
    assert_eq!(log4_int_64(16), 2);
}
