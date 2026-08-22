use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::parser::common::parse_params;
use pbrt_r4::parser::{parse_string, DebugTarget};

#[test]
fn parse_string_accepts_comments_between_arguments_and_operations() {
    let mut target = DebugTarget::new();
    parse_string("Translate 1 # the first component\n 2 3\n", &mut target)
        .expect("comments should be accepted by the nom parser");

    let operations = target.operations.borrow();
    let names: Vec<&str> = operations
        .iter()
        .map(|operation| operation.name.as_str())
        .collect();
    assert_eq!(names, ["Translate"]);
}

#[test]
fn parse_params_accepts_comments_inside_arrays() {
    let (_, params): (&str, ParameterDictionary) =
        parse_params("\"float values\" [1 # between values\n 2 3]").unwrap();

    assert_eq!(params.get_floats("values"), vec![1.0, 2.0, 3.0]);
}

#[test]
fn parse_params_accepts_comments_between_name_and_value() {
    let (_, params): (&str, ParameterDictionary) =
        parse_params("\"float value\" # before the value\n [1]").unwrap();

    assert_eq!(params.get_floats("value"), vec![1.0]);
}

#[test]
fn parse_string_accepts_comment_only_lines() {
    let mut target = DebugTarget::new();
    parse_string("# a comment-only line\nIdentity\n", &mut target)
        .expect("comment-only lines should be ignored");

    let operations = target.operations.borrow();
    assert_eq!(operations.len(), 1);
}

#[test]
fn hash_inside_string_is_not_a_comment() {
    let (_, params): (&str, ParameterDictionary) =
        parse_params("\"string label\" [\"# not a comment\"]").unwrap();

    assert_eq!(params.get_strings("label"), vec!["# not a comment"]);
}

#[test]
fn parse_string_accepts_comment_at_eof() {
    let mut target = DebugTarget::new();
    parse_string("Identity # no trailing newline", &mut target)
        .expect("a comment at EOF should be accepted");

    let operations = target.operations.borrow();
    assert_eq!(operations.len(), 1);
}
