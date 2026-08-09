use pbrt_r4::util::sampling::iterators::*;

#[test]
fn iterator_lengths_match_v4_generators() {
    assert_eq!(Uniform1D::new(4, 0).count(), 4);
    assert_eq!(Uniform2D::new(4, 0).count(), 4);
    assert_eq!(Uniform3D::new(4, 0).count(), 4);
    assert_eq!(Stratified1D::new(4, 0).count(), 4);
    assert_eq!(Stratified2D::new(2, 3, 0).count(), 6);
    assert_eq!(Stratified3D::new(2, 2, 3, 0).count(), 12);
    assert_eq!(Hammersley2D::new(5).count(), 5);
    assert_eq!(Hammersley3D::new(5).count(), 5);
}

#[test]
fn stratified_samples_stay_in_unit_domain() {
    for p in Stratified2D::new(4, 3, 7) {
        assert!((0.0..1.0).contains(&p.x));
        assert!((0.0..1.0).contains(&p.y));
    }
    for p in Stratified3D::new(2, 2, 2, 7) {
        assert!((0.0..1.0).contains(&p.x));
        assert!((0.0..1.0).contains(&p.y));
        assert!((0.0..1.0).contains(&p.z));
    }
}

#[test]
fn hammersley_first_coordinate_matches_v4() {
    let samples: Vec<_> = Hammersley2D::new(4).collect();
    assert_eq!(samples[0].x, 0.0);
    assert_eq!(samples[1].x, 0.25);
    assert_eq!(samples[2].x, 0.5);
    assert_eq!(samples[3].x, 0.75);
    assert!(samples.iter().all(|p| (0.0..1.0).contains(&p.y)));
}
