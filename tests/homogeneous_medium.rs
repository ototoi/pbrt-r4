use pbrt_r4::media::HomogeneousMedium;
use pbrt_r4::util::base::{Float, Point3f, Vector3f};
use pbrt_r4::util::geometry::ray::Ray;
use pbrt_r4::util::spectrum::*;

#[test]
fn sample_point_matches_coefficients() {
    let sigma_a = Spectrum::from([0.01, 0.02, 0.03]);
    let sigma_s = Spectrum::from([0.04, 0.05, 0.06]);
    let medium = HomogeneousMedium::new(&sigma_a, &sigma_s, 1.0, &Spectrum::zero(), 1.0, 0.35);
    let lambda = SampledWavelengths::sample_visible(0.37);
    let mp = medium.sample_point(&Point3f::new(1.0, 2.0, 3.0), &lambda);
    assert!(mp.sigma_a.max_component_value() > 0.0);
    assert!(mp.sigma_s.max_component_value() > 0.0);
    assert_eq!(mp.le, SampledSpectrum::zero());
}

#[test]
fn sample_ray_returns_single_majorant_segment() {
    let sigma_a = Spectrum::from([0.1, 0.2, 0.3]);
    let sigma_s = Spectrum::from([0.4, 0.5, 0.6]);
    let medium = HomogeneousMedium::new(&sigma_a, &sigma_s, 1.0, &Spectrum::zero(), 1.0, 0.0);
    let ray = Ray::new(
        &Point3f::zero(),
        &Vector3f::new(0.0, 0.0, 1.0),
        Float::INFINITY,
        0.0,
    );
    let lambda = SampledWavelengths::sample_visible(0.37);
    let mut iter = medium.sample_ray(&ray, 2.5, &lambda).unwrap();
    let seg = iter.next().unwrap();
    assert_eq!(seg.t_min, 0.0);
    assert_eq!(seg.t_max, 2.5);
    assert!(seg.sigma_maj.max_component_value() > 0.0);
    assert!(iter.next().is_none());
}

#[test]
fn wavelength_sampling_uses_dense_coefficients() {
    let medium = HomogeneousMedium::new_with_sources(
        Spectrum::from_rgb(&[0.1, 0.2, 0.7], SpectrumType::Albedo),
        blackbody_spectrum(&vec![5500.0, 0.25]),
        0.0,
    );
    let lambda = SampledWavelengths::sample_visible(0.37);
    let mp = medium.sample_point(&Point3f::zero(), &lambda);
    assert!(mp.sigma_a.max_component_value() > 0.0);
    assert!(mp.sigma_s.max_component_value() > 0.0);
}
