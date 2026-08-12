use pbrt_r4::ext::skymodel::HosekSkyModel;

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1e-10 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
    );
}

#[test]
fn spectral_wavelengths_match_v4_model_bands() {
    assert_eq!(
        HosekSkyModel::spectral_wavelengths(),
        &[320.0, 360.0, 400.0, 440.0, 480.0, 520.0, 560.0, 600.0, 640.0, 680.0, 720.0]
    );
}

#[test]
fn solar_radiance_matches_v4_representative_points() {
    let model = HosekSkyModel::new(10.0_f64.to_radians(), 3.0, 0.5).unwrap();
    let points = [
        ((0.0, 0.0, 320.0), 2358.0656127426628),
        ((0.4, 0.7, 400.0), 0.036312395544706125),
        ((1.0, 2.0, 560.0), 0.03757406901683466),
        ((1.4, 0.2, 720.0), 0.24354046181253097),
    ];

    for ((theta, gamma, wavelength), expected) in points {
        let actual = model.solar_radiance(theta, gamma, wavelength).unwrap();
        assert_close(actual, expected);
    }
}

#[test]
fn model_rejects_invalid_state_inputs_without_fallback() {
    assert!(HosekSkyModel::new(-0.001, 3.0, 0.5).is_err());
    assert!(HosekSkyModel::new(0.0, 0.99, 0.5).is_err());
    assert!(HosekSkyModel::new(0.0, 3.0, 1.01).is_err());

    let model = HosekSkyModel::new(0.0, 3.0, 0.5).unwrap();
    assert!(model.solar_radiance(0.0, 0.0, 319.9).is_err());
    assert_eq!(model.radiance(0.0, 0.0, 279.9), 0.0);
}
