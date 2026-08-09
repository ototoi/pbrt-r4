use pbrt_r4::util::base::Point2f;
use pbrt_r4::util::sampling::piecewise_linear_2d::PiecewiseLinear2D;

#[test]
fn flat_density_warp_is_identity() {
    let data = vec![1.0f32; 4 * 4];
    let pl: PiecewiseLinear2D<0> = PiecewiseLinear2D::new(&data, 4, 4, [], [], true, true);
    let r = pl.sample(Point2f::new(0.5, 0.5), &[]);
    assert!((r.p.x - 0.5).abs() < 5e-3, "x={}", r.p.x);
    assert!((r.p.y - 0.5).abs() < 5e-3, "y={}", r.p.y);
    assert!((r.pdf - 1.0).abs() < 5e-3, "pdf={}", r.pdf);
}

#[test]
fn sample_invert_round_trip_no_conditioning() {
    let data = vec![
        0.1f32, 0.2, 0.4, 0.8, 0.2, 0.3, 0.5, 0.9, 0.4, 0.5, 0.6, 1.0, 0.8, 0.9, 1.0, 1.2,
    ];
    let pl: PiecewiseLinear2D<0> = PiecewiseLinear2D::new(&data, 4, 4, [], [], true, true);

    for &u in &[
        Point2f::new(0.13, 0.27),
        Point2f::new(0.51, 0.49),
        Point2f::new(0.78, 0.85),
    ] {
        let warped = pl.sample(u, &[]);
        let recovered = pl.invert(warped.p, &[]);
        assert!(
            (recovered.p.x - u.x).abs() < 1e-3,
            "u.x={}, recovered.x={}",
            u.x,
            recovered.p.x
        );
        assert!(
            (recovered.p.y - u.y).abs() < 1e-3,
            "u.y={}, recovered.y={}",
            u.y,
            recovered.p.y
        );
        let pdf_eval = pl.evaluate(warped.p, &[]);
        assert!(
            (warped.pdf - recovered.pdf).abs() < 1e-3 * warped.pdf.max(1e-6),
            "pdf mismatch: warp={} invert={}",
            warped.pdf,
            recovered.pdf
        );
        assert!(
            (warped.pdf - pdf_eval).abs() < 1e-3 * warped.pdf.max(1e-6),
            "pdf vs evaluate mismatch: warp={} evaluate={}",
            warped.pdf,
            pdf_eval
        );
    }
}

#[test]
fn one_parameter_value_collapses_to_unconditioned() {
    let data = vec![0.5f32; 9];
    let pl_no_param: PiecewiseLinear2D<0> = PiecewiseLinear2D::new(&data, 3, 3, [], [], true, true);
    let theta = [0.0f32];
    let pl_one_param: PiecewiseLinear2D<1> =
        PiecewiseLinear2D::new(&data, 3, 3, [1], [&theta], true, true);
    let p = Point2f::new(0.4, 0.6);
    let a = pl_no_param.evaluate(p, &[]);
    let b = pl_one_param.evaluate(p, &[0.0]);
    assert!((a - b).abs() < 1e-6, "a={}, b={}", a, b);
}
