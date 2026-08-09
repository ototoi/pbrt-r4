use pbrt_r4::util::spectrum::named::{lookup_named_spectrum, lookup_named_spectrum_curve};

#[test]
fn named_spectrum_lookup_is_case_sensitive_like_v4() {
    assert!(lookup_named_spectrum("glass-BK7").is_some());
    assert!(lookup_named_spectrum("glass-bk7").is_none());
    assert!(lookup_named_spectrum_curve("metal-Ag-eta").is_some());
    assert!(lookup_named_spectrum_curve("metal-ag-eta").is_none());
}
