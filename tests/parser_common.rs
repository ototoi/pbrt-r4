use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::parser::common::parse_params;
use pbrt_r4::parser::parsed_parameter::into_parameter_dictionary;
use pbrt_r4::util::spectrum::Spectrum;

#[test]
fn parse_numeric_spectrum_pairs() {
    let (_, params) = parse_params("\"spectrum sigma_s\" [200 10 900 10]").unwrap();
    let params: ParameterDictionary = into_parameter_dictionary(params);
    let sp = params.get_one_spectrum("sigma_s", &Spectrum::zero());
    assert_eq!(sp, Spectrum::from_sampled(&[200.0, 900.0], &[10.0, 10.0]));
    let sampled = params.get_sampled_spectra_ref("sigma_s").unwrap();
    assert_eq!(sampled[0].lambda, vec![200.0, 900.0]);
    assert_eq!(sampled[0].values, vec![10.0, 10.0]);
}

#[test]
fn parse_string_spectrum_remains_string() {
    let (_, params) = parse_params("\"spectrum eta\" \"metal-Cu-eta\"").unwrap();
    let params = into_parameter_dictionary(params);
    assert_eq!(params.get_strings("eta"), vec!["metal-Cu-eta"]);
}

#[test]
fn parse_integer_returns_error_on_invalid_input() {
    assert!(parse_params("\"integer xresolution\" [maybe]").is_err());
}
