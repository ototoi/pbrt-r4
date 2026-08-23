use pbrt_r4::parser::{parse_string, DebugTarget};

fn operation_names(target: &DebugTarget) -> Vec<String> {
    target
        .operations
        .borrow()
        .iter()
        .map(|operation| operation.name.clone())
        .collect()
}

#[test]
fn dispatch_preserves_common_prefix_directives() {
    let mut target = DebugTarget::new();
    parse_string(
        "AttributeBegin\nAttributeEnd\nTransformBegin\nTransformEnd\nWorldBegin\nWorldEnd\n",
        &mut target,
    )
    .expect("common-prefix directives should be dispatched exactly");

    assert_eq!(
        operation_names(&target),
        [
            "AttributeBegin",
            "AttributeEnd",
            "TransformBegin",
            "TransformEnd",
            "WorldBegin",
            "WorldEnd",
        ]
    );
}

#[test]
fn dispatch_accepts_comment_after_directive_name() {
    let mut target = DebugTarget::new();
    parse_string(
        "WorldBegin# comment without a separating space\n",
        &mut target,
    )
    .expect("a comment may immediately follow a void directive");

    assert_eq!(operation_names(&target), ["WorldBegin"]);
}

#[test]
fn dispatch_rejects_a_known_name_with_a_suffix() {
    let mut target = DebugTarget::new();
    let error = parse_string("AttributeBeginExtra\n", &mut target)
        .expect_err("a suffixed directive must not match AttributeBegin");

    assert!(error.msg.contains("operation `AttributeBeginExtra`"));
    assert!(target.operations.borrow().is_empty());
}
