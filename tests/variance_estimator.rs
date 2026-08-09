use pbrt_r4::util::sampling::variance_estimator::VarianceEstimator;

#[test]
fn matches_v4_online_variance_and_merge() {
    let mut all = VarianceEstimator::default();
    for value in [1.0, 2.0, 3.0, 4.0] {
        all.add(value);
    }
    assert_eq!(all.count(), 4);
    assert!((all.mean() - 2.5).abs() < 1e-6);
    assert!((all.variance() - 1.6666666).abs() < 1e-5);

    let mut left = VarianceEstimator::default();
    let mut right = VarianceEstimator::default();
    left.add(1.0);
    left.add(2.0);
    right.add(3.0);
    right.add(4.0);
    left.merge(&right);
    assert!((left.mean() - all.mean()).abs() < 1e-6);
    assert!((left.variance() - all.variance()).abs() < 1e-6);
}
