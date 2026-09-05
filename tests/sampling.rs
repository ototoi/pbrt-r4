use pbrt_r4::util::sampling::*;

#[test]
fn stratified_sample_2d_supports_non_square_grids() {
    let mut rng = RNG::new_sequence(7);
    let mut samples = vec![Point2f::zero(); 6];
    stratified_sample_2d(&mut samples, 3, 2, &mut rng, false);

    assert_eq!(samples.len(), 6);
    for (index, sample) in samples.iter().enumerate() {
        let x = index % 3;
        let y = index / 3;
        assert!((sample.x - (x as Float + 0.5) / 3.0).abs() < 1e-6);
        assert!((sample.y - (y as Float + 0.5) / 2.0).abs() < 1e-6);
    }
}

#[test]
fn balance_heuristic_matches_v4_definition() {
    assert!((balance_heuristic(1, 0.25, 1, 0.75) - 0.25).abs() < 1e-6);
    assert_eq!(balance_heuristic(1, 0.0, 1, 0.0), 0.0);
}

#[test]
fn discrete_sampling_clamps_upper_boundary_like_v4() {
    let sample = sample_discrete(&[1.0, 2.0, 4.0], 1.0).unwrap();
    assert_eq!(sample.0, 2);
    assert!(sample.2 <= ONE_MINUS_EPSILON);
}

#[test]
fn continuous_distribution_samples_invert_like_v4() {
    for u in [0.1, 0.5, 0.9] {
        let x = sample_exponential(u, 2.0);
        assert!((invert_exponential_sample(x, 2.0) - u).abs() < 1e-6);
        let x = sample_normal(u, 1.0, 2.0);
        assert!((invert_normal_sample(x, 1.0, 2.0) - u).abs() < 1e-5);
        let x = sample_logistic(u, 3.0);
        assert!((invert_logistic_sample(x, 3.0) - u).abs() < 1e-6);
        let x = sample_trimmed_exponential(u, 2.0, 4.0);
        assert!((invert_trimmed_exponential_sample(x, 2.0, 4.0) - u).abs() < 1e-6);
        let x = sample_smooth_step(u, -2.0, 5.0);
        assert!((invert_smooth_step_sample(x, -2.0, 5.0) - u).abs() < 1e-5);
        let x = sample_trimmed_logistic(u, 3.0, -4.0, 5.0);
        assert!((invert_trimmed_logistic_sample(x, 3.0, -4.0, 5.0) - u).abs() < 1e-6);
    }
}

#[test]
fn geometric_sampling_inverts_like_v4() {
    let u = Point2f::new(0.23, 0.71);
    let disk = uniform_sample_disk_polar(&u);
    let recovered = invert_uniform_disk_polar_sample(&disk);
    assert!((recovered.x - u.x).abs() < 1e-5);
    assert!((recovered.y - u.y).abs() < 1e-5);

    let disk = concentric_sample_disk(&u);
    let recovered = invert_uniform_disk_concentric_sample(&disk);
    assert!((recovered.x - u.x).abs() < 1e-5);
    assert!((recovered.y - u.y).abs() < 1e-5);

    let hemisphere = uniform_sample_hemisphere(&u);
    let recovered = invert_uniform_hemisphere_sample(&hemisphere);
    assert!((recovered.x - u.x).abs() < 1e-5);
    assert!((recovered.y - u.y).abs() < 1e-5);

    let sphere = uniform_sample_sphere(&u);
    let recovered = invert_uniform_sphere_sample(&sphere);
    assert!((recovered.x - u.x).abs() < 1e-5);
    assert!((recovered.y - u.y).abs() < 1e-5);

    let cosine = cosine_sample_hemisphere(&u);
    let recovered = invert_cosine_hemisphere_sample(&cosine);
    assert!((recovered.x - u.x).abs() < 1e-5);
    assert!((recovered.y - u.y).abs() < 1e-5);

    let cone = uniform_sample_cone(&u, 0.4);
    let recovered = invert_uniform_cone_sample(&cone, 0.4);
    assert!((recovered.x - u.x).abs() < 1e-5);
    assert!((recovered.y - u.y).abs() < 1e-5);
}

#[test]
fn linear_and_tent_sampling_match_v4() {
    for u in [0.1, 0.5, 0.9] {
        let x = sample_linear(u, 1.0, 4.0);
        assert!((invert_linear_sample(x, 1.0, 4.0) - u).abs() < 1e-5);
        let x = sample_tent(u, 2.0);
        assert!((invert_tent_sample(x, 2.0) - u).abs() < 1e-5);
        assert!(tent_pdf(x, 2.0) > 0.0);
    }
    assert_eq!(linear_pdf(-0.1, 1.0, 4.0), 0.0);
}

#[test]
fn henyey_greenstein_sampling_returns_v4_pdf() {
    let (wi, pdf) =
        sample_henyey_greenstein(Vector3f::new(0.0, 0.0, 1.0), 0.5, Point2f::new(0.3, 0.7));
    assert!((wi.length() - 1.0).abs() < 1e-5);
    assert!(pdf.is_finite() && pdf > 0.0);
}

#[test]
fn spherical_rectangle_sampling_returns_v4_solid_angle_pdf() {
    let u = Point2f::new(0.3, 0.7);
    let (p, pdf) = sample_spherical_rectangle(
        Point3f::new(0.0, 0.0, 2.0),
        Point3f::new(-1.0, -1.0, 0.0),
        Vector3f::new(2.0, 0.0, 0.0),
        Vector3f::new(0.0, 2.0, 0.0),
        u,
    );
    assert!(p.z.abs() < 1e-6);
    assert!(pdf.is_finite() && pdf > 0.0);
    let recovered = invert_spherical_rectangle_sample(
        Point3f::new(0.0, 0.0, 2.0),
        Point3f::new(-1.0, -1.0, 0.0),
        Vector3f::new(2.0, 0.0, 0.0),
        Vector3f::new(0.0, 2.0, 0.0),
        p,
    );
    assert!(
        (recovered.x - u.x).abs() < 2e-3,
        "u={:?} recovered={:?}",
        u,
        recovered
    );
    assert!(
        (recovered.y - u.y).abs() < 2e-3,
        "u={:?} recovered={:?}",
        u,
        recovered
    );
}

#[test]
fn spherical_triangle_sampling_and_inverse_agree() {
    let triangle = [
        Point3f::new(-1.0, -0.5, 0.0),
        Point3f::new(1.0, -0.5, 0.0),
        Point3f::new(0.0, 1.0, 0.0),
    ];
    let reference = Point3f::new(0.2, 0.1, 2.0);

    for u in [
        Point2f::new(0.1, 0.2),
        Point2f::new(0.4, 0.7),
        Point2f::new(0.8, 0.3),
    ] {
        let (barycentrics, pdf) = sample_spherical_triangle(&triangle, reference, u);
        let b = barycentrics.expect("the non-degenerate triangle must be sampleable");
        let sampled_point = triangle[0] * b[0] + triangle[1] * b[1] + triangle[2] * b[2];
        let wi = (sampled_point - reference).normalize();
        let recovered = invert_spherical_triangle_sample(&triangle, reference, wi);

        assert!(pdf.is_finite() && pdf > 0.0);
        assert!((b.iter().sum::<Float>() - 1.0).abs() < 1e-5);
        assert!(
            (recovered.x - u.x).abs() < 2e-4,
            "u={u:?}, recovered={recovered:?}"
        );
        assert!(
            (recovered.y - u.y).abs() < 2e-4,
            "u={u:?}, recovered={recovered:?}"
        );
    }
}

#[test]
fn bilinear_warp_sampling_and_pdf_match_v4_reference() {
    let weights = [0.1, 0.5, 1.5, 3.0];
    for u in [
        Point2f::new(0.1, 0.2),
        Point2f::new(0.4, 0.7),
        Point2f::new(0.8, 0.3),
    ] {
        let warped = sample_bilinear(u, &weights);
        let recovered = invert_bilinear_sample(warped, &weights);
        assert!((recovered.x - u.x).abs() < 1e-5);
        assert!((recovered.y - u.y).abs() < 1e-5);
        assert!(bilinear_pdf(warped, &weights).is_finite());
        assert!(bilinear_pdf(warped, &weights) > 0.0);
    }
}

#[test]
fn catmull_rom_sampling_returns_v4_value_and_pdf() {
    let result =
        sample_catmull_rom(&[0.0, 1.0, 2.0], &[1.0, 2.0, 1.0], &[0.0, 1.5, 3.0], 0.5).unwrap();
    assert!(result.0 > 0.0 && result.0 < 2.0);
    assert!(result.1.is_finite() && result.1 > 0.0);
    assert!(result.2.is_finite() && result.2 > 0.0);
}

#[test]
fn discrete_sampling_matches_v4_zero_weight_rules() {
    assert_eq!(sample_discrete(&[], 0.5), None);
    let sample = sample_discrete(&[0.0, 2.0, 1.0], 0.1).unwrap();
    assert_eq!(sample.0, 1);
    assert!((sample.1 - 2.0 / 3.0).abs() < 1e-6);
    assert!(sample.2 < 1.0);
}
