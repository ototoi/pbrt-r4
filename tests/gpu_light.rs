use pbrt_r4::gpu::webgpu::light::{area_pdf_omega, uniform_light_pmf};

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
