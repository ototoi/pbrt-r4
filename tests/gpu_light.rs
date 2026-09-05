use pbrt_r4::gpu::webgpu::light::{
    area_pdf_omega, area_triangle_pmf, triangle_world_area, uniform_light_pmf,
};

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
fn area_triangle_pmf_matches_uniform_area_pdf() {
    let areas = [1.0, 3.0];
    let total_area = areas.into_iter().sum::<f32>();
    let pmfs = areas.map(|area| area_triangle_pmf(area, total_area));

    assert!((pmfs.into_iter().sum::<f32>() - 1.0).abs() < 1e-6);
    for (area, pmf) in areas.into_iter().zip(pmfs) {
        let conditional_pdf = 1.0 / area;
        assert!((pmf * conditional_pdf - 1.0 / total_area).abs() < 1e-6);
    }
}

#[test]
fn area_triangle_pmf_rejects_invalid_measurements() {
    assert_eq!(area_triangle_pmf(0.0, 1.0), 0.0);
    assert_eq!(area_triangle_pmf(1.0, 0.0), 0.0);
    assert_eq!(area_triangle_pmf(f32::NAN, 1.0), 0.0);
    assert_eq!(area_triangle_pmf(1.0, f32::INFINITY), 0.0);
}
