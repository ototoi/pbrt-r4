use pbrt_r4::util::imageio::ColorEncoding;

#[test]
fn color_encoding_parses_linear_and_srgb() {
    assert_eq!(
        ColorEncoding::parse("linear").unwrap(),
        ColorEncoding::Linear
    );
    assert_eq!(ColorEncoding::parse("sRGB").unwrap(), ColorEncoding::SRgb);
}

#[test]
fn color_encoding_parses_gamma_value() {
    assert_eq!(
        ColorEncoding::parse("gamma 2.2").unwrap(),
        ColorEncoding::Gamma(2.2)
    );
}

#[test]
fn color_encoding_rejects_invalid_gamma() {
    assert!(ColorEncoding::parse("gamma").is_err());
    assert!(ColorEncoding::parse("gamma 0").is_err());
    assert!(ColorEncoding::parse("gamma -1").is_err());
    assert!(ColorEncoding::parse("unknown").is_err());
}

#[test]
fn color_encoding_legacy_bool_maps_to_linear_or_srgb() {
    assert_eq!(
        ColorEncoding::from_legacy_gamma(false),
        ColorEncoding::Linear
    );
    assert_eq!(ColorEncoding::from_legacy_gamma(true), ColorEncoding::SRgb);
}

#[test]
fn linear_encoding_round_trips_byte_values() {
    let encoding = ColorEncoding::Linear;
    for value in [0.0, 0.25, 0.5, 1.0] {
        assert_eq!(encoding.to_linear(encoding.from_linear(value)), value);
    }
}

#[test]
fn srgb_encoding_is_not_linear_at_midpoint() {
    let encoding = ColorEncoding::SRgb;
    let encoded = encoding.from_linear(0.5);
    assert!(encoded > 0.5);
    assert!((encoding.to_linear(encoded) - 0.5).abs() < 1e-5);
}

#[test]
fn gamma_encoding_does_not_clamp_float_input_before_power() {
    let value = ColorEncoding::Gamma(2.2).to_linear(-0.5);
    assert!(value.is_nan());
}
