use pbrt_r4::media::CloudMedium;
use pbrt_r4::prelude::*;

#[test]
fn cloud_density_stays_within_its_bounds() {
    let sigma_a = Spectrum::from(0.01);
    let sigma_s = Spectrum::from(0.5);
    let medium = CloudMedium::new(
        &Bounds3f::new(&Point3f::zero(), &Point3f::new(1.0, 1.0, 1.0)),
        &Transform::identity(),
        &sigma_a,
        &sigma_s,
        0.0,
        2.0,
        1.0,
        5.0,
    );
    let lambda = SampledWavelengths::sample_visible(0.37);
    for p in [Point3f::new(0.5, 0.5, 0.5), Point3f::new(0.2, 1.1, 0.2)] {
        let properties = medium.sample_point(&p, &lambda);
        assert!(properties.sigma_a.max_component_value() >= 0.0);
        assert!(properties.sigma_s.max_component_value() >= 0.0);
        assert!(properties.sigma_a.max_component_value() <= 0.01);
        assert!(properties.sigma_s.max_component_value() <= 0.5);
    }
}
