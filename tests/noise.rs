use pbrt_r4::textures::noise::{noise, turbulence};
use pbrt_r4::util::base::{Float, Point3f, Vector3f};

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

#[test]
fn turbulence_adds_clamped_octaves_like_v4() {
    // With huge footprints no octave is evaluated (n = 0). pbrt-v4 then adds
    // Lerp(SmoothStep(0), 0.2, |noise|) = 0.2 for the partial octave plus 0.2
    // per clamped octave weighted by omega^i.
    let p = Point3f::new(0.3, 0.7, 0.1);
    let big = Vector3f::new(10.0, 0.0, 0.0);
    let omega = 0.5;
    let expected = 0.2 + 0.2 * (1.0 + 0.5 + 0.25 + 0.125);
    let value = turbulence(&p, &big, &big, omega, 4);
    assert!((value - expected).abs() < 1e-6, "{value} != {expected}");
}
