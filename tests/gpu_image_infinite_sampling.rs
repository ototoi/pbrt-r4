use pbrt_r4::gpu::flat::ImageInfiniteDistribution;

#[test]
fn compensated_distribution_samples_a_bright_pixel_and_reports_its_pdf() {
    let rgb = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [9.0, 9.0, 9.0],
    ];
    let distribution = ImageInfiniteDistribution::from_rgb(2, 2, &rgb).unwrap();

    let uv = distribution.sample([0.75, 0.4]).unwrap();

    assert!((uv[0] - 0.875).abs() < 1e-6);
    assert!((uv[1] - 0.7).abs() < 1e-6);
    assert!((distribution.pdf(uv) - 4.0).abs() < 1e-6);
}

#[test]
fn compensated_distribution_is_uniform_for_black_and_constant_images() {
    for rgb in [vec![[0.0; 3]; 6], vec![[2.0; 3]; 6]] {
        let distribution = ImageInfiniteDistribution::from_rgb(3, 2, &rgb).unwrap();
        let u = [0.17, 0.83];
        let uv = distribution.sample(u).unwrap();

        assert!((uv[0] - u[0]).abs() < 1e-6);
        assert!((uv[1] - u[1]).abs() < 1e-6);
        assert!((distribution.pdf(uv) - 1.0).abs() < 1e-6);
    }
}

#[test]
fn compensation_uses_the_v4_mean_before_clipping_negative_values() {
    let values = [-10.0, 1.0, 2.0, 2.0];
    let rgb = values.map(|value| [value; 3]);
    let distribution = ImageInfiniteDistribution::from_rgb(2, 2, &rgb).unwrap();

    let weights = distribution
        .texels
        .iter()
        .map(|texel| texel.weight)
        .collect::<Vec<_>>();
    assert_eq!(weights, [0.0, 2.25, 3.25, 3.25]);
}

#[test]
fn zero_weight_rows_are_never_sampled_and_pdf_integrates_to_one() {
    let rgb = [[0.0; 3], [0.0; 3], [0.0; 3], [3.0; 3]];
    let distribution = ImageInfiniteDistribution::from_rgb(2, 2, &rgb).unwrap();

    for y in 0..8 {
        for x in 0..8 {
            let u = [(x as f32 + 0.5) / 8.0, (y as f32 + 0.5) / 8.0];
            let uv = distribution.sample(u).unwrap();
            assert!(uv[0] >= 0.5 && uv[1] >= 0.5);
            assert!((distribution.pdf(uv) - 4.0).abs() < 1e-6);
        }
    }

    let rgb = [[-10.0; 3], [1.0; 3], [2.0; 3], [2.0; 3]];
    let distribution = ImageInfiniteDistribution::from_rgb(2, 2, &rgb).unwrap();
    let integrated_pdf = distribution
        .texels
        .iter()
        .map(|texel| texel.weight / distribution.total_integral / 4.0)
        .sum::<f32>();
    assert!((integrated_pdf - 1.0).abs() < 1e-6);
}
