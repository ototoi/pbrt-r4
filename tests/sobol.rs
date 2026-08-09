use pbrt_r4::util::base::{Float, Point2i};
use pbrt_r4::util::lowdiscrepancy::sobol::sobol::{sobol_interval_to_index, sobol_sample};

#[test]
fn interval_to_index_samples_land_in_target_pixel() {
    for log_res in 0..8 {
        let res = 1 << log_res;
        for y in 0..res {
            for x in 0..res {
                let mut saw_corner = false;
                for s in 0..16 {
                    let index = sobol_interval_to_index(log_res, s, &Point2i::new(x, y));
                    let sx = sobol_sample(index as i64, 0, 0) * res as Float;
                    let sy = sobol_sample(index as i64, 1, 0) * res as Float;
                    if sx == x as Float && sy == y as Float {
                        assert!(
                            !saw_corner,
                            "multiple lower-left corner samples for pixel ({x},{y}), res {res}, sample {s}"
                        );
                        saw_corner = true;
                    }
                    let ix = sx as i32;
                    let iy = sy as i32;
                    assert!(
                        (ix == x && iy == y)
                            || (x == ix && sy == (y + 1) as Float)
                            || (sx == (x + 1) as Float && y == iy),
                        "log_res {log_res}, res {res}, pixel ({x},{y}), sample {s}, got ({sx},{sy}), index {index}"
                    );
                }
            }
        }
    }
}
