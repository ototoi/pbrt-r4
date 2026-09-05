use pbrt_r4::gpu::webgpu::light::{area_pdf_omega, triangle_world_area, uniform_light_pmf};

#[test]
fn uniform_light_pmf_is_normalized() {
    let pmf = uniform_light_pmf(4);
    assert!((4.0 * pmf - 1.0).abs() < 1e-6);
    assert_eq!(uniform_light_pmf(0), 0.0);
}

#[test]
fn area_pdf_omega_uses_solid_angle_measure() {
    assert!((area_pdf_omega(4.0, 0.5, 2.0) - 4.0).abs() < 1e-6);
    assert_eq!(area_pdf_omega(1.0, 0.0, 1.0), 0.0);
    assert_eq!(area_pdf_omega(f32::NAN, 1.0, 1.0), 0.0);
}

#[test]
fn triangle_area_is_computed_after_the_instance_transform() {
    let positions = [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ];
    let transform = [
        2.0, 0.0, 0.0, 4.0, 0.0, 3.0, 0.0, 5.0, 0.0, 0.0, 7.0, 6.0, 0.0, 0.0, 0.0, 1.0,
    ];

    assert_eq!(triangle_world_area(transform, positions), Some(3.0));
}

#[test]
fn per_triangle_light_sampling_accounts_for_unequal_areas() {
    let pmf = uniform_light_pmf(2);
    let areas = [1.0, 3.0];
    let expected_integral = areas.into_iter().sum::<f32>();
    let sampled_estimates = areas.map(|area| 1.0 / (pmf * (1.0 / area)));
    let expectation = pmf * sampled_estimates.into_iter().sum::<f32>();

    assert!((expectation - expected_integral).abs() < 1e-6);
}
