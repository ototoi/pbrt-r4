use pbrt_r4::textures::noise::noise;
use pbrt_r4::util::base::{Float, Point3f};

#[test]
fn noise_handles_large_coordinates_like_v4() {
    let period = (1_i32 << 30) as Float;
    let base = Point3f::new(0.5, -1.0, 2.0);
    let shifted = Point3f::new(base.x + period, base.y - period, base.z + 2.0 * period);

    assert_eq!(
        noise(base.x, base.y, base.z),
        noise(shifted.x, shifted.y, shifted.z)
    );
}
