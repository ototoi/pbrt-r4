use pbrt_r4::util::spectrum::named::{lookup_named_spectrum, lookup_named_spectrum_curve};
use pbrt_r4::util::spectrum::source::spectrum_from_named;
use pbrt_r4::util::spectrum::{VISIBLE_LAMBDA_MAX, VISIBLE_LAMBDA_MIN};

#[test]
fn named_spectrum_lookup_is_case_sensitive_like_v4() {
    assert!(lookup_named_spectrum("glass-BK7").is_some());
    assert!(lookup_named_spectrum("glass-bk7").is_none());
    assert!(lookup_named_spectrum_curve("metal-Ag-eta").is_some());
    assert!(lookup_named_spectrum_curve("metal-ag-eta").is_none());
}

#[test]
fn named_spectrum_source_extends_endpoint_values_to_the_visible_range() {
    let spectrum = spectrum_from_named("glass-F11").unwrap();

    assert!(spectrum.sample_at(VISIBLE_LAMBDA_MIN) > 0.0);
    assert!(spectrum.sample_at(VISIBLE_LAMBDA_MAX) > 0.0);
}
