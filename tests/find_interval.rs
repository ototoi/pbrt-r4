// Imported from find_interval.cpp

use pbrt_r4::util::base::*;

#[test]
fn find_interval_basics() {
    let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // Check clamping for out of range
    assert_eq!(0, find_interval(&a, &|x, index| x[index] < -1.0));
    assert_eq!(
        a.len() - 2,
        find_interval(&a, &|x, index| x[index] <= 100.0)
    );

    for i in 0..a.len() - 1 {
        assert_eq!(i, find_interval(&a, &|x, index| x[index] <= i as Float));
        assert_eq!(
            i,
            find_interval(&a, &|x, index| x[index] < i as Float + 0.5)
        );
        if i > 0 {
            assert_eq!(
                i - 1,
                find_interval(&a, &|x, index| x[index] <= i as Float - 0.5)
            );
        }
    }
}
