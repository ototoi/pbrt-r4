use std::f32::consts::PI;

use pbrt_r4::util::transform::interval::{sqr, Interval};

#[test]
fn normalizes_bounds_and_reports_geometry() {
    let i = Interval::new(4.0, 2.0);
    assert_eq!(i.low, 2.0);
    assert_eq!(i.high, 4.0);
    assert_eq!(i.midpoint(), 3.0);
    assert_eq!(i.width(), 2.0);
    assert!(i.contains(3.0));
}

#[test]
fn division_containing_zero_is_unbounded() {
    let i = Interval::new(1.0, 2.0) / Interval::new(-1.0, 1.0);
    assert!(i.low.is_infinite() && i.low.is_sign_negative());
    assert!(i.high.is_infinite() && i.high.is_sign_positive());
}

#[test]
fn square_handles_zero_crossing() {
    let crossing = sqr(Interval::new(-2.0, 3.0));
    assert_eq!(crossing.low, 0.0);
    assert!(crossing.high >= 9.0);
    let positive = sqr(Interval::new(2.0, 3.0));
    assert!(positive.low <= 4.0 && positive.high >= 9.0);
}

#[test]
fn value_and_error_expands_bounds() {
    let i = Interval::from_value_and_error(2.0, 0.25);
    assert!(i.low < 1.75);
    assert!(i.high > 2.25);
}

#[test]
fn trigonometric_intervals_include_v4_extrema() {
    let sine = Interval::sin(Interval::new(0.0, PI));
    assert_eq!(sine.high, 1.0);

    let cosine = Interval::cos(Interval::new(0.0, 2.0 * PI));
    assert_eq!(cosine.low, -1.0);
    assert!(cosine.high >= 1.0);
}

#[test]
fn acos_clamps_and_expands_bounds() {
    let result = Interval::acos(Interval::new(-2.0, 2.0));
    assert_eq!(result.low, 0.0);
    assert!(result.high >= PI);
}
