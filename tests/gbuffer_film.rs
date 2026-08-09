use pbrt_r4::film::gbuffer_film::normalize_gbuffer_normal;

#[test]
fn normalize_gbuffer_normal_matches_v4_writeout() {
    let n = normalize_gbuffer_normal([0.0, 0.0, 4.0]);
    assert_eq!(n, [0.0, 0.0, 1.0]);

    let tilted = normalize_gbuffer_normal([3.0, 4.0, 0.0]);
    assert!((tilted[0] - 0.6).abs() < 1e-6);
    assert!((tilted[1] - 0.8).abs() < 1e-6);
    assert_eq!(tilted[2], 0.0);

    assert_eq!(normalize_gbuffer_normal([0.0; 3]), [0.0; 3]);
}
