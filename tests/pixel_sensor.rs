use pbrt_r4::film::pixel_sensor::PixelSensor;
use pbrt_r4::util::spectrum::{Spectrum, SpectrumType};

#[test]
fn unknown_sensor_is_rejected() {
    assert!(PixelSensor::create("not-a-real-sensor", 100.0, 0.0).is_err());
}

#[test]
fn cie1931_sensor_iso_scales_output_linearly() {
    let s100 = PixelSensor::create("cie1931", 100.0, 0.0).unwrap();
    let s150 = PixelSensor::create("cie1931", 150.0, 0.0).unwrap();
    let spectrum = Spectrum::from_rgb(&[0.2, 0.4, 0.6], SpectrumType::Albedo);
    let rgb100 = s100.to_output_rgb(&spectrum);
    let rgb150 = s150.to_output_rgb(&spectrum);
    for c in 0..3 {
        assert!(rgb100[c].is_finite() && rgb150[c].is_finite());
        assert!((rgb150[c] - rgb100[c] * 1.5).abs() < 1e-4);
    }
}

#[test]
fn canon_sensor_white_swatch_round_trip() {
    let sensor = PixelSensor::create("canon_eos_5d_mkiv", 100.0, 0.0).unwrap();
    let spectrum = Spectrum::from(1.0);
    let rgb = sensor.to_output_rgb(&spectrum);
    for c in rgb.iter() {
        assert!(c.is_finite(), "non-finite canon output: {:?}", rgb);
    }
    let max = rgb[0].abs().max(rgb[1].abs()).max(rgb[2].abs());
    assert!(max > 0.0, "canon output should be non-trivially positive");
}
