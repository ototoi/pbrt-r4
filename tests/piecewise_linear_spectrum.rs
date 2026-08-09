use pbrt_r4::util::base::Float;
use pbrt_r4::util::spectrum::config::{LAMBDA_MAX, LAMBDA_MIN};
use pbrt_r4::util::spectrum::piecewise_linear::PiecewiseLinearSpectrum;

#[test]
fn from_interleaved_extends_to_visible_range() {
    let samples = [400.0, 2.0, 500.0, 3.0, 600.0, 4.0];
    let spectrum = PiecewiseLinearSpectrum::from_interleaved(&samples, false).unwrap();
    assert_eq!(
        spectrum.lambda.first().copied(),
        Some(LAMBDA_MIN as Float - 1.0)
    );
    assert_eq!(spectrum.values.first().copied(), Some(2.0));
    assert_eq!(
        spectrum.lambda.last().copied(),
        Some(LAMBDA_MAX as Float + 1.0)
    );
    assert_eq!(spectrum.values.last().copied(), Some(4.0));
}

#[test]
fn from_interleaved_normalize_matches_v4_intent() {
    let samples = [400.0, 1.0, 500.0, 1.0, 600.0, 1.0, 700.0, 1.0];
    let spectrum = PiecewiseLinearSpectrum::from_interleaved(&samples, true).unwrap();
    let y = spectrum.evaluate().y();
    assert!(
        (y - 1.0).abs() < 1e-4,
        "expected normalized luminance, got {y}"
    );
}
