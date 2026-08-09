use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::util::base::Point3f;
use pbrt_r4::util::spectrum::Spectrum;

#[test]
fn typed_values_accept_prefixed_and_unprefixed_names() {
    let mut params = ParameterDictionary::new();
    params.add_bool("bool is_a", true);
    assert!(params.get_one_bool("is_a", false));
    assert!(params.get_one_bool("bool is_a", false));
    assert!(!params.get_one_bool("fuga", false));
}

#[test]
fn scalar_and_point_defaults_are_applied() {
    let mut params = ParameterDictionary::new();
    params.add_int("integer count", 1234);
    assert_eq!(params.get_one_int("count", 5678), 1234);
    assert_eq!(params.get_one_int("missing", 5678), 5678);
    let p = Point3f::from([1.0, 2.0, 3.0]);
    let d = Point3f::from([4.0, 5.0, 6.0]);
    params.add_point3f("point P", &p);
    assert_eq!(params.get_one_point3f("P", &d), p);
    assert_eq!(params.get_one_point3f("missing", &d), d);
}

#[test]
fn string_and_spectrum_values_use_expected_defaults() {
    let mut params = ParameterDictionary::new();
    params.add_string("string value", "hello!");
    assert_eq!(params.get_one_string("value", "world!"), "hello!");
    let value = Spectrum::from([1.0, 2.0, 3.0]);
    params.add_spectrum("spectrum value", &value);
    assert_eq!(params.get_one_spectrum("value", &Spectrum::zero()), value);
    assert_eq!(params.get_one_spectrum("missing", &value), value);
}

#[test]
fn clone_and_set_copy_parameter_values() {
    let mut params = ParameterDictionary::new();
    params.add_int("integer count", 1234);
    let mut other = params.clone();
    other.add_int("integer total", 78910);
    assert_eq!(params.get_one_int("total", 4567), 4567);
    params.set(&other);
    assert_eq!(params.get_one_int("total", 4567), 78910);
}

#[test]
fn spectrum_values_support_float_and_rgb_inputs() {
    let mut params = ParameterDictionary::new();
    params.add_float("float s", 2.5);
    assert_eq!(
        params.get_one_spectrum("s", &Spectrum::zero()),
        Spectrum::from(2.5)
    );
    params.add_rgb("rgb sig", &[1.0, 2.0, 3.0, 0.25, 0.5, 0.75]);
    assert_eq!(params.get_spectrum_array("sig").len(), 2);
}
