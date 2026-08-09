use pbrt_r4::base::Filter;
use pbrt_r4::prelude::*;

fn make_u(i: u32) -> Point2f {
    // Deterministic pseudo-random samples in [0,1).
    let a = i.wrapping_mul(1664525).wrapping_add(1013904223);
    let b = i.wrapping_mul(22695477).wrapping_add(1);
    let u0 = (a as Float) / (u32::MAX as Float + 1.0);
    let u1 = (b as Float) / (u32::MAX as Float + 1.0);
    Point2f::new(u0, u1)
}

fn make_radius_params(xr: Float, yr: Float) -> ParameterDictionary {
    let mut params = ParameterDictionary::new();
    params.add_float("xradius", xr);
    params.add_float("yradius", yr);
    params
}

#[test]
fn filter_create_uses_radius_params() {
    let params = make_radius_params(1.25, 0.75);
    let names = ["box", "gaussian", "mitchell", "sinc", "triangle"];

    for name in names {
        let f =
            Filter::create(name, &params).unwrap_or_else(|_| panic!("failed to create {}", name));
        let r = f.radius();
        assert!((r.x - 1.25).abs() < 1e-6);
        assert!((r.y - 0.75).abs() < 1e-6);
    }
}

#[test]
fn filter_sample_is_in_radius_and_finite() {
    let filters = [
        ("box", make_radius_params(1.0, 1.0)),
        ("gaussian", {
            let mut p = make_radius_params(1.5, 1.5);
            p.add_float("sigma", 0.5);
            p
        }),
        ("mitchell", make_radius_params(2.0, 2.0)),
        ("sinc", make_radius_params(4.0, 4.0)),
        ("triangle", make_radius_params(2.0, 2.0)),
    ];

    for (name, params) in filters {
        let f =
            Filter::create(name, &params).unwrap_or_else(|_| panic!("failed to create {}", name));
        let r = f.radius();
        for i in 0..1024 {
            let u = make_u(i);
            let s = f.sample(&u);
            assert!(
                s.p.x.is_finite() && s.p.y.is_finite(),
                "non-finite sample point for {}",
                name
            );
            assert!(
                s.weight.is_finite(),
                "non-finite sample weight for {}",
                name
            );
            assert!(
                s.p.x.abs() <= r.x + 1e-6,
                "sample x out of radius for {}",
                name
            );
            assert!(
                s.p.y.abs() <= r.y + 1e-6,
                "sample y out of radius for {}",
                name
            );
        }
    }
}

#[test]
fn filter_integral_is_positive() {
    let filters = [
        ("box", make_radius_params(1.0, 1.0)),
        ("gaussian", {
            let mut p = make_radius_params(1.5, 1.5);
            p.add_float("sigma", 0.5);
            p
        }),
        ("mitchell", make_radius_params(2.0, 2.0)),
        ("sinc", make_radius_params(4.0, 4.0)),
        ("triangle", make_radius_params(2.0, 2.0)),
    ];

    for (name, params) in filters {
        let f =
            Filter::create(name, &params).unwrap_or_else(|_| panic!("failed to create {}", name));
        assert!(f.integral() > 0.0, "non-positive integral for {}", name);
    }
}

#[test]
fn signed_filter_sampling_preserves_negative_lobes() {
    let f = Filter::create("sinc", &make_radius_params(4.0, 4.0))
        .expect("failed to create sinc filter");

    let mut saw_negative = false;
    for i in 0..4096 {
        let s = f.sample(&make_u(i));
        saw_negative |= s.weight < 0.0;
    }

    assert!(
        saw_negative,
        "sinc filter samples should keep signed weights"
    );
}
