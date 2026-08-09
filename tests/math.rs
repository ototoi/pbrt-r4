use pbrt_r4::util::base::Float;
use pbrt_r4::util::math::{
    evaluate_polynomial, fast_exp, is_power_of_four, log2, log2_int, log4_int, log_i0,
    newton_bisection, round_up_power_of_four, safe_acos, safe_asin,
};

#[test]
fn v4_scalar_helpers_match_expected_values() {
    assert_eq!(log2_int(1), 0);
    assert_eq!(log2_int(8), 3);
    assert_eq!(log4_int(1), 0);
    assert_eq!(log4_int(4), 1);
    assert_eq!(log4_int(63), 2);
    assert!(is_power_of_four(64));
    assert!(!is_power_of_four(32));
    assert_eq!(round_up_power_of_four(65), 256);
    assert!((fast_exp(1.0) - (1.0 as Float).exp()).abs() < 0.01);
    assert_eq!(safe_asin(1.00005), std::f64::consts::FRAC_PI_2 as Float);
    assert_eq!(safe_acos(-1.00005), std::f64::consts::PI as Float);
    assert!((log2(8.0) - 3.0).abs() < 1e-6);
    assert_eq!(evaluate_polynomial(2.0, &[1.0, 2.0, 3.0]), 17.0);
    assert!((log_i0(0.0)).abs() < 1e-6);
    let root = newton_bisection(0.0, 2.0, |x| (x * x - 2.0, 2.0 * x));
    assert!((root - Float::sqrt(2.0)).abs() < 1e-5);
}
