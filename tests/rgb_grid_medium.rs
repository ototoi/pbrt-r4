use pbrt_r4::media::RGBGridMedium;
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::prelude::*;

#[test]
fn rgb_grid_sample_point_uses_sigma_grids() {
    let mut parameters = ParameterDictionary::new();
    parameters.add_int("nx", 2);
    parameters.add_int("ny", 1);
    parameters.add_int("nz", 1);
    parameters.add_rgb("sigma_a", &[0.1, 0.2, 0.3, 0.2, 0.3, 0.4]);
    parameters.add_rgb("sigma_s", &[0.4, 0.5, 0.6, 0.5, 0.6, 0.7]);
    let medium = RGBGridMedium::create(&parameters, &Transform::identity()).unwrap();

    let lambda = SampledWavelengths::sample_visible(0.37);
    let mp = medium.sample_point(&Point3f::new(0.25, 0.5, 0.5), &lambda);
    assert!(mp.sigma_a.max_component_value() > 0.0);
    assert!(mp.sigma_s.max_component_value() > 0.0);
    assert_eq!(mp.le, SampledSpectrum::zero());
}

#[test]
fn rgb_grid_create_uses_illuminant_le_grid() {
    let mut parameters = ParameterDictionary::new();
    parameters.add_rgb("rgb sigma_a", &[0.1, 0.2, 0.3]);
    parameters.add_rgb("rgb Le", &[1.0, 0.5, 0.25]);
    let Ok(medium) = RGBGridMedium::create(&parameters, &Transform::identity()) else {
        panic!("rgbgrid with sigma_a and Le should be valid");
    };

    let lambda = SampledWavelengths::sample_visible(0.37);
    let mp = medium.sample_point(&Point3f::new(0.5, 0.5, 0.5), &lambda);
    assert!(medium.is_emissive());
    assert!(mp.le.max_component_value() > 0.0);
}

#[test]
fn rgb_grid_sample_ray_returns_majorant_grid_segments() {
    let mut parameters = ParameterDictionary::new();
    parameters.add_int("nx", 1);
    parameters.add_int("ny", 1);
    parameters.add_int("nz", 1);
    parameters.add_rgb("sigma_a", &[0.1, 0.2, 0.3]);
    let medium = RGBGridMedium::create(&parameters, &Transform::identity()).unwrap();
    let ray = Ray::new(
        &Point3f::new(0.5, 0.5, -1.0),
        &Vector3f::new(0.0, 0.0, 1.0),
        Float::INFINITY,
        0.0,
    );

    let lambda = SampledWavelengths::sample_visible(0.37);
    let Some(mut iter) = medium.sample_ray(&ray, Float::INFINITY, &lambda) else {
        panic!("rgb grid medium should yield a DDA iterator");
    };
    let Some(seg) = iter.next() else {
        panic!("rgb grid medium should yield at least one segment");
    };
    assert!(seg.t_max > seg.t_min);
}
