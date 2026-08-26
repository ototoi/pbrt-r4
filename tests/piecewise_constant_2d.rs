use pbrt_r4::util::base::Point2f;
use pbrt_r4::util::geometry::Bounds2f;
use pbrt_r4::util::sampling::piecewise_constant_2d::{PiecewiseConstant1D, PiecewiseConstant2D};

#[test]
fn one_dimensional_sampling_respects_domain_at_u_one() {
    let distribution = PiecewiseConstant1D::new(&[1.0, 1.0], -2.0, 4.0);
    let (value, pdf, remapped) = distribution.sample(1.0);
    assert!(value < 4.0 && value > 3.99);
    assert!((pdf - 1.0 / 6.0).abs() < 1e-6);
    assert!(remapped < 1.0);
}

#[test]
fn two_dimensional_sampling_respects_domain() {
    let domain = Bounds2f::new(&Point2f::new(-2.0, 3.0), &Point2f::new(2.0, 7.0));
    let distribution = PiecewiseConstant2D::new_with_domain(&[1.0, 2.0, 3.0, 4.0], 2, 2, domain);
    let (p, pdf) = distribution.sample_continuous(&Point2f::new(0.25, 0.75));
    assert!(domain.inside(&p));
    assert!(pdf > 0.0);
    assert_eq!(distribution.pdf(&Point2f::new(-3.0, 5.0)), 0.0);
}

#[test]
fn domain_width_contributes_to_integral() {
    let distribution = PiecewiseConstant1D::new(&[2.0, 2.0], -2.0, 4.0);
    assert!((distribution.integral() - 12.0).abs() < 1e-6);
}

#[test]
fn one_dimensional_pdf_is_normalized_on_non_unit_domain() {
    let distribution = PiecewiseConstant1D::new(&[1.0, 1.0], -2.0, 4.0);
    assert!((distribution.pdf(0.0) - 1.0 / 6.0).abs() < 1e-6);
}

#[test]
fn one_dimensional_sample_invert_round_trips() {
    let distribution = PiecewiseConstant1D::new(&[1.0, 3.0], -2.0, 4.0);
    for sample in [0.0, 0.125, 0.25, 0.5, 0.875, 1.0] {
        let (value, _, _) = distribution.sample(sample);
        let inverse = distribution.invert(value).unwrap();
        assert!((inverse - sample.min(1.0 - 1e-7)).abs() < 1e-6);
    }
}
