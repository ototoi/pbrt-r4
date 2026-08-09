use pbrt_r4::prelude::*;

#[test]
fn medium_homogeneous_defaults_to_unit_sigma_and_zero_le() {
    let params = ParameterDictionary::new();
    let t = Transform::identity();
    let medium = Medium::create("homogeneous", &params, &t).unwrap();
    match medium {
        Medium::Homogeneous(m) => {
            let lambda = pbrt_r4::util::spectrum::SampledWavelengths::sample_visible(0.37);
            let mp = m.sample_point(&Point3f::zero(), &lambda);
            assert_eq!(mp.sigma_a, SampledSpectrum::from(1.0));
            assert_eq!(mp.sigma_s, SampledSpectrum::from(1.0));
            assert_eq!(mp.le, SampledSpectrum::zero());
        }
        _ => panic!("expected homogeneous medium"),
    }
}

#[test]
fn medium_homogeneous_unknown_preset_falls_back_to_defaults_like_v4() {
    let mut params = ParameterDictionary::new();
    params.add_string("preset", "not-a-real-medium");
    let t = Transform::identity();
    let medium = Medium::create("homogeneous", &params, &t).unwrap();
    match medium {
        Medium::Homogeneous(m) => {
            let lambda = pbrt_r4::util::spectrum::SampledWavelengths::sample_visible(0.37);
            let mp = m.sample_point(&Point3f::zero(), &lambda);
            assert_eq!(mp.sigma_a, SampledSpectrum::from(1.0));
            assert_eq!(mp.sigma_s, SampledSpectrum::from(1.0));
            assert_eq!(mp.le, SampledSpectrum::zero());
        }
        _ => panic!("expected homogeneous medium"),
    }
}

#[test]
fn medium_grid_defaults_to_unit_sigma_and_zero_emission() {
    let mut params = ParameterDictionary::new();
    params.add_int("nx", 1);
    params.add_int("ny", 1);
    params.add_int("nz", 1);
    params.add_floats("density", &[1.0]);
    let mut explicit = ParameterDictionary::new();
    explicit.add_int("nx", 1);
    explicit.add_int("ny", 1);
    explicit.add_int("nz", 1);
    explicit.add_floats("density", &[1.0]);
    explicit.add_floats("sigma_a", &[1.0]);
    explicit.add_floats("sigma_s", &[1.0]);
    let t = Transform::identity();
    let medium = Medium::create("uniformgrid", &params, &t).unwrap();
    let explicit = Medium::create("uniformgrid", &explicit, &t).unwrap();
    match medium {
        Medium::Grid(m) => {
            if let Medium::Grid(explicit) = explicit {
                let lambda = pbrt_r4::util::spectrum::SampledWavelengths::sample_visible(0.37);
                let p = Point3f::zero();
                let a = m.sample_point(&p, &lambda);
                let b = explicit.sample_point(&p, &lambda);
                assert_eq!(a.sigma_a, b.sigma_a);
                assert_eq!(a.sigma_s, b.sigma_s);
                assert_eq!(a.le, b.le);
            } else {
                panic!("expected grid medium");
            }
        }
        _ => panic!("expected grid medium"),
    }
}

#[test]
fn medium_cloud_defaults_to_unit_sigma_density_one_and_unit_frequency() {
    let params = ParameterDictionary::new();
    let mut explicit = ParameterDictionary::new();
    explicit.add_float("density", 1.0);
    explicit.add_float("wispiness", 1.0);
    explicit.add_float("frequency", 5.0);
    let t = Transform::identity();
    let medium = Medium::create("cloud", &params, &t).unwrap();
    let explicit = Medium::create("cloud", &explicit, &t).unwrap();
    match medium {
        Medium::Cloud(m) => {
            if let Medium::Cloud(explicit) = explicit {
                let lambda = pbrt_r4::util::spectrum::SampledWavelengths::sample_visible(0.37);
                let p = Point3f::new(0.1, 0.2, 0.3);
                let a = m.sample_point(&p, &lambda);
                let b = explicit.sample_point(&p, &lambda);
                assert_eq!(a.sigma_a, b.sigma_a);
                assert_eq!(a.sigma_s, b.sigma_s);
                assert_eq!(a.le, b.le);
            } else {
                panic!("expected cloud medium");
            }
        }
        _ => panic!("expected cloud medium"),
    }
}

#[test]
fn medium_nanovdb_defaults_to_density_grid_and_temperature_grid_names() {
    let params = ParameterDictionary::new();
    let t = Transform::identity();
    let medium = Medium::create("nanovdb", &params, &t);
    assert!(medium.is_err());
}
