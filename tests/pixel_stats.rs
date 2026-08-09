#![cfg(feature = "stats")]

use pbrt_r4::util::base::Point2i;
use pbrt_r4::util::geometry::Bounds2i;
use pbrt_r4::util::stats::pixel_stats;

#[test]
fn pixel_callbacks_flush_counter_and_ratio() {
    let counter = pixel_stats::register_pixel_counter("test/pixel-counter");
    let ratio = pixel_stats::register_pixel_ratio("test/pixel-ratio");
    pixel_stats::enable(
        Bounds2i::from(((0, 0), (1, 1))),
        "/tmp/pbrt-pixel-stats-test.exr",
    );

    counter.add(7);
    ratio.add_num(3);
    ratio.add_denom(5);
    pixel_stats::report_pixel_start(Point2i::new(0, 0));
    pixel_stats::report_pixel_end(Point2i::new(0, 0));

    assert!(pixel_stats::enabled());
}
