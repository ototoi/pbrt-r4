use pbrt_r4::util::spectrum::source::{canonical_spectrum_type, SpectrumType};

#[test]
fn unknown_spectrum_class_is_not_remapped() {
    assert_eq!(canonical_spectrum_type("unknown"), None);
    assert_eq!(
        canonical_spectrum_type("reflectance"),
        Some(SpectrumType::Albedo)
    );
}
