use pbrt_r4::paramdict::find_type_from_key;

#[test]
fn well_known_parameter_type_is_available() {
    assert_eq!(find_type_from_key("xresolution"), Some("integer"));
}
