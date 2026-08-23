use pbrt_r4::parser::common::parse_params;
use pbrt_r4::parser::parsed_parameter::{
    into_parameter_dictionary, parsed_parameters_from_dictionary, ParsedParameterValues,
    ParsedSpectrumToken,
};
use pbrt_r4::parser::{parse_string, DebugTarget, ToPlyTarget};
use std::cell::RefCell;
use std::sync::Arc;

#[test]
fn parse_params_keeps_owned_typed_values() {
    let (_, parameters) = parse_params(
        "\"bool enabled\" true \"integer count\" [1 2] \"float scale\" 0.5 \"string label\" \"hair\"",
    )
    .expect("parameters should parse");

    assert!(matches!(
        &parameters[0].values,
        ParsedParameterValues::Bools(values) if values == &[true]
    ));
    assert!(matches!(
        &parameters[1].values,
        ParsedParameterValues::Ints(values) if values == &[1, 2]
    ));
    assert!(matches!(
        &parameters[2].values,
        ParsedParameterValues::Floats(values) if values == &[0.5]
    ));
    assert!(matches!(
        &parameters[3].values,
        ParsedParameterValues::Strings(values) if values == &["hair"]
    ));
}

#[test]
fn spectrum_tokens_preserve_numeric_and_named_paths() {
    let (_, numeric) = parse_params("\"spectrum sigma\" [200 10 900 20]").unwrap();
    assert!(matches!(
        &numeric[0].values,
            ParsedParameterValues::SpectrumTokens(values)
            if values == &[
                ParsedSpectrumToken::Float { value: 200.0, raw: "200".to_string() },
                ParsedSpectrumToken::Float { value: 10.0, raw: "10".to_string() },
                ParsedSpectrumToken::Float { value: 900.0, raw: "900".to_string() },
                ParsedSpectrumToken::Float { value: 20.0, raw: "20".to_string() },
            ]
    ));

    let (_, named) = parse_params("\"spectrum eta\" \"metal-Cu-eta\"").unwrap();
    assert!(matches!(
        &named[0].values,
        ParsedParameterValues::SpectrumTokens(values)
            if values == &[ParsedSpectrumToken::String("metal-Cu-eta".to_string())]
    ));
}

#[test]
fn mixed_spectrum_tokens_preserve_numeric_lexemes() {
    let (_, parameters) = parse_params("\"spectrum values\" [1e2 \"named\"]").unwrap();
    let dictionary = into_parameter_dictionary(parameters);

    assert_eq!(dictionary.get_strings("values"), vec!["1e2", "named"]);
}

#[test]
fn parsed_parameters_convert_to_the_existing_dictionary_storage() {
    let (_, parameters) = parse_params("\"rgb reflectance\" [0.2 0.3 0.4]").unwrap();
    let dictionary = into_parameter_dictionary(parameters);

    assert_eq!(dictionary.get_points("reflectance"), vec![0.2, 0.3, 0.4]);
}

#[test]
fn dictionary_round_trip_preserves_xyz_storage() {
    let (_, parameters) = parse_params("\"xyz value\" [0.1 0.2 0.3]").unwrap();
    let dictionary = into_parameter_dictionary(parameters);
    let round_trip = into_parameter_dictionary(parsed_parameters_from_dictionary(&dictionary));

    assert_eq!(
        round_trip.get_points("value"),
        dictionary.get_points("value")
    );
}

#[test]
fn dictionary_round_trip_preserves_untyped_well_known_points() {
    let (_, parameters) = parse_params("\"P\" [1 2 3]").unwrap();
    let dictionary = into_parameter_dictionary(parameters);
    let round_trip = into_parameter_dictionary(parsed_parameters_from_dictionary(&dictionary));

    assert_eq!(round_trip.get_points("P"), dictionary.get_points("P"));
}

#[test]
fn dictionary_adapter_preserves_empty_typed_storage() {
    let mut dictionary = pbrt_r4::paramdict::ParameterDictionary::new();
    dictionary.add_owned_strings("string empty", Vec::new());

    let parameters = parsed_parameters_from_dictionary(&dictionary);

    assert!(matches!(
        &parameters[0].values,
        ParsedParameterValues::Strings(values) if values.is_empty()
    ));
}

#[test]
fn to_ply_forwards_small_triangle_parameters() {
    let directory = tempfile::tempdir().unwrap();
    let inner = Arc::new(RefCell::new(DebugTarget::new()));
    let mut target = ToPlyTarget::new(directory.path().to_str().unwrap(), inner.clone());
    parse_string(
        "Shape \"trianglemesh\" \"integer indices\" [0 1 2] \"point3 P\" [0 0 0 1 0 0 0 1 0]",
        &mut target,
    )
    .unwrap();

    assert!(inner
        .borrow()
        .operations
        .borrow()
        .iter()
        .any(|operation| operation.name == "Shape"));
}

#[test]
fn invalid_typed_values_are_rejected_without_partial_parameters() {
    assert!(parse_params("\"integer count\" [1 nope]").is_err());
}
