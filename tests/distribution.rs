use pbrt_r4::util::base::Point2f;
use pbrt_r4::util::geometry::Bounds2f;
use pbrt_r4::util::sampling::distribution::{Distribution1D, Distribution2D};

#[test]
fn discrete_sampling_handles_zero_weight_intervals() {
    let distribution = Distribution1D::new(&[0.0, 1.0, 0.0]);
    let (index, _, remapped) = distribution.sample_discrete(0.0);
    assert_eq!(index, 1);
    assert_eq!(remapped, 0.0);
}

#[test]
fn negative_function_values_match_v4_absolute_sampling() {
    let distribution = Distribution1D::new(&[-2.0, 1.0]);
    assert_eq!(distribution.func, vec![2.0, 1.0]);
    assert!((distribution.discrete_pdf(0) - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn zero_integral_discrete_pdf_is_zero() {
    let distribution = Distribution1D::new(&[0.0, 0.0]);
    assert_eq!(distribution.discrete_pdf(0), 0.0);
}

#[test]
fn continuous_sampling_respects_custom_domain() {
    let distribution = Distribution1D::new_with_domain(&[1.0, 1.0], 2.0, 4.0);
    let (sample, pdf, _) = distribution.sample_continuous(0.5);
    assert!((sample - 3.0).abs() < 1e-6);
    assert!((pdf - 0.5).abs() < 1e-6);
    assert!((distribution.invert(sample).unwrap() - 0.5).abs() < 1e-6);
}

#[test]
fn distribution2d_sample_invert_round_trips() {
    let distribution = Distribution2D::new_with_domain(
        &[1.0, 2.0, 3.0, 4.0],
        2,
        2,
        Bounds2f::from(((2.0, 4.0), (6.0, 8.0))),
    );
    let (point, _) = distribution.sample_continuous(&Point2f::new(0.25, 0.75));
    let inverse = distribution.invert(&point).unwrap();
    assert!((inverse.x - 0.25).abs() < 1e-6);
    assert!((inverse.y - 0.75).abs() < 1e-6);
}

#[test]
fn one_dimensional_boundaries_match_v4() {
    let distribution = Distribution1D::new(&[1.0, 2.0, 4.0]);

    let (at_start, _, start_offset) = distribution.sample_continuous(0.0);
    let (at_end, _, end_offset) = distribution.sample_continuous(1.0);
    assert_eq!(at_start, 0.0);
    assert_eq!(at_end, 1.0);
    assert_eq!(start_offset, 0);
    assert_eq!(end_offset, 2);
    assert_eq!(distribution.invert(0.0), Some(0.0));
    assert_eq!(distribution.invert(1.0), Some(1.0));
}

#[test]
fn two_dimensional_pdf_clamps_to_edge_and_rejects_zero_integral() {
    let distribution = Distribution2D::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert_eq!(distribution.pdf(&Point2f::new(-1.0, 2.0)), 1.2);
    assert_eq!(distribution.pdf(&Point2f::new(2.0, -1.0)), 0.8);

    let zero = Distribution2D::new(&[0.0, 0.0, 0.0, 0.0], 2, 2);
    assert_eq!(zero.pdf(&Point2f::new(0.5, 0.5)), 0.0);
}
