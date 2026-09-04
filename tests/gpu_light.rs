use pbrt_r4::gpu::webgpu::light::{area_pdf_omega, build_triangle_cdf, uniform_light_pmf};

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
fn triangle_cdf_excludes_degenerate_faces_and_ends_at_one() {
    let (total, cdf) = build_triangle_cdf(&[(3, 1.0), (4, 0.0), (5, 3.0)]).unwrap();
    assert_eq!(total, 4.0);
    assert_eq!(cdf.len(), 2);
    assert_eq!(cdf[0].0, 3);
    assert_eq!(cdf[1], (5, 1.0));
    assert!(cdf[0].1 > 0.0 && cdf[0].1 < cdf[1].1);
}
