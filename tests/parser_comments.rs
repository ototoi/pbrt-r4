use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::parser::common::parse_params;
use pbrt_r4::parser::{parse_string, parse_string_upgraded, DebugTarget, SceneBuilder};

fn debug_operations(target: &DebugTarget) -> Vec<(String, Vec<String>)> {
    target
        .operations
        .borrow()
        .iter()
        .map(|operation| (operation.name.clone(), operation.args.clone()))
        .collect()
}

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
fn parse_string_evaluates_operations_in_order() {
    let scene = "Identity\nTranslate 1 2 3\nScale 2 2 2\n";
    let mut streaming_target = DebugTarget::new();
    let mut ast_target = DebugTarget::new();
    parse_string(scene, &mut streaming_target).expect("streaming parse should succeed");
    parse_string_upgraded(scene, &mut ast_target).expect("AST parse should succeed");

    assert_eq!(
        debug_operations(&streaming_target),
        debug_operations(&ast_target)
    );
}

#[test]
fn parse_string_rejects_an_invalid_operation_after_valid_input() {
    let mut target = DebugTarget::new();
    let result = parse_string("Identity\nNotAParserOperation\n", &mut target);

    let error = result.expect_err("invalid operation should be rejected");
    assert!(error.msg.contains("line 2"));
    assert!(error.msg.contains("operation `NotAParserOperation`"));
    assert_eq!(debug_operations(&target).len(), 1);
}

#[test]
fn streaming_and_ast_paths_keep_large_shape_counts() {
    const SHAPE_COUNT: usize = 1024;
    let mut scene = String::from("WorldBegin\n");
    for _ in 0..SHAPE_COUNT {
        scene.push_str("Shape \"sphere\" \"float radius\" [1]\n");
    }
    scene.push_str("WorldEnd\n");

    let mut streaming_builder = SceneBuilder::new();
    let mut ast_builder = SceneBuilder::new();
    parse_string(&scene, &mut streaming_builder).expect("streaming parse should succeed");
    parse_string_upgraded(&scene, &mut ast_builder).expect("AST parse should succeed");

    assert_eq!(streaming_builder.shapes.len(), SHAPE_COUNT);
    assert_eq!(streaming_builder.shapes.len(), ast_builder.shapes.len());
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
