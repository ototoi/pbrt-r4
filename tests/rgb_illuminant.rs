use pbrt_r4::util::spectrum::rgb_illuminant::RGBIlluminantSpectrum;
use pbrt_r4::util::spectrum::rgb_to_spectrum::ACES2065_1;
use pbrt_r4::util::spectrum::sampled::SampledWavelengths;

#[test]
fn from_color_space_uses_color_space_illuminant() {
    let lambda = SampledWavelengths::sample_visible(0.31);
    let rgb = [1.0, 1.0, 1.0];
    let spectrum = RGBIlluminantSpectrum::from_color_space(&ACES2065_1, rgb);
    let expected = ACES2065_1.illuminant_to_sampled_spectrum(rgb, &lambda);

    assert_eq!(spectrum.sample(&lambda), expected);
    assert_eq!(
        spectrum.illuminant_dense().sample_at(560.0),
        ACES2065_1.illuminant.sample_at(560.0)
    );
}
