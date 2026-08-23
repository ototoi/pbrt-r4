use pbrt_r4::parser::{parse_string, parse_string_upgraded, DebugTarget};

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

#[test]
fn dispatches_every_parser_operation() {
    let cases = [
        "Identity\n",
        "Translate 1 2 3\n",
        "Rotate 1 0 1 0\n",
        "Scale 1 1 1\n",
        "LookAt 0 0 1 0 0 0 0 1 0\n",
        "ConcatTransform [1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1]\n",
        "Transform [1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1]\n",
        "CoordinateSystem \"camera\"\n",
        "CoordSysTransform \"camera\"\n",
        "ColorSpace \"srgb\"\n",
        "ActiveTransform All\n",
        "PixelFilter \"box\"\n",
        "Film \"rgb\"\n",
        "Sampler \"independent\"\n",
        "Accelerator \"bvh\"\n",
        "Integrator \"path\"\n",
        "Camera \"perspective\"\n",
        "MakeNamedMedium \"medium\"\n",
        "MediumInterface \"inside\" \"outside\"\n",
        "Option \"integer xresolution\" 1\n",
        "WorldBegin\n",
        "Attribute \"material\"\n",
        "AttributeBegin\n",
        "AttributeEnd\n",
        "TransformBegin\n",
        "TransformEnd\n",
        "Texture \"texture\" \"spectrum\" \"imagemap\"\n",
        "Material \"diffuse\"\n",
        "MakeNamedMaterial \"material\"\n",
        "NamedMaterial \"material\"\n",
        "LightSource \"point\"\n",
        "AreaLightSource \"diffuse\"\n",
        "Shape \"sphere\"\n",
        "ReverseOrientation\n",
        "ObjectBegin \"object\"\n",
        "ObjectEnd\n",
        "ObjectInstance \"object\"\n",
        "WorldEnd\n",
        "Include \"child.pbrt\"\n",
        "Import \"child.pbrt\"\n",
        "WorkDirBegin \"/tmp\"\n",
        "WorkDirEnd\n",
    ];

    for source in cases {
        let mut target = DebugTarget::new();
        parse_string(source, &mut target)
            .unwrap_or_else(|error| panic!("failed to dispatch {source:?}: {error:?}"));
        assert_eq!(
            target.operations.borrow().len(),
            1,
            "expected one operation for {source:?}"
        );
    }
}

#[test]
fn dispatch_reports_invalid_identifier_and_unknown_operation() {
    for source in ["123\n", "Shape-é\n"] {
        let mut target = DebugTarget::new();
        let error = parse_string(source, &mut target).expect_err("invalid name should fail");
        assert!(
            error.msg.contains("line 1"),
            "unexpected error: {}",
            error.msg
        );
        assert!(target.operations.borrow().is_empty());
    }

    let mut target = DebugTarget::new();
    let error =
        parse_string("UnknownDirective\n", &mut target).expect_err("unknown directive should fail");
    assert!(error.msg.contains("operation `UnknownDirective`"));
}

#[test]
fn dispatch_preserves_malformed_operation_errors() {
    let mut target = DebugTarget::new();
    let error = parse_string("Translate 1 2\n", &mut target)
        .expect_err("missing operation argument should fail");

    assert!(error.msg.contains("line 1"));
    assert!(error.msg.contains("operation `Translate`"));
}

#[test]
fn dispatch_streaming_and_upgrade_paths_have_the_same_operations() {
    let source = "WorldBegin\nAttributeBegin\nShape \"sphere\"\nAttributeEnd\nWorldEnd\n";
    let mut streaming = DebugTarget::new();
    let mut upgraded = DebugTarget::new();

    parse_string(source, &mut streaming).expect("streaming parser should succeed");
    parse_string_upgraded(source, &mut upgraded).expect("upgrade parser should succeed");

    assert_eq!(operation_names(&streaming), operation_names(&upgraded));
}
