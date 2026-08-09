use pbrt_r4::util::spectrum::{spectrum_to_photometric, Spectrum, SpectrumType};

#[test]
fn rgb_illuminant_photometric_value_ignores_rgb_multiplier() {
    // pbrt-v4's SpectrumToPhotometric uses only the underlying illuminant for
    // RGBIlluminantSpectrum. The RGB multiplier is applied separately by the
    // light when computing target power or illuminance.
    let base = Spectrum::from_rgb_illuminant(&[1.0, 1.0, 1.0]);
    let scaled = Spectrum::from_rgb_illuminant(&[0.2, 0.4, 0.8]);

    let base_photometric = spectrum_to_photometric(&base);
    let scaled_photometric = spectrum_to_photometric(&scaled);

    assert!(base_photometric > 0.0);
    assert!((base_photometric - scaled_photometric).abs() < 1e-5);
}

#[test]
fn non_illuminant_photometric_value_scales_with_spectrum() {
    let spectrum = Spectrum::from_rgb(&[0.2, 0.4, 0.6], SpectrumType::Unbounded);
    let doubled = Spectrum::from_rgb(&[0.4, 0.8, 1.2], SpectrumType::Unbounded);

    let value = spectrum_to_photometric(&spectrum);
    let doubled_value = spectrum_to_photometric(&doubled);

    assert!(value > 0.0);
    assert!((doubled_value - 2.0 * value).abs() < 1e-5);
}
