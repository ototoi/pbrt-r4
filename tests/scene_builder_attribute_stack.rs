use pbrt_r4::parser::{parse_string, SceneBuilder};
use pbrt_r4::util::transform::Transform;

#[test]
fn attribute_end_restores_transform_for_following_shapes() {
    let mut builder = SceneBuilder::new();
    let scene = r#"
WorldBegin
AttributeBegin
Translate 1 2 3
Shape "sphere"
AttributeEnd
Shape "sphere"
WorldEnd
"#;

    parse_string(scene, &mut builder).expect("scene should parse");
    assert_eq!(builder.shapes.len(), 2);
    assert_ne!(
        builder.shapes[0].render_from_object.primary(),
        Transform::identity()
    );
    assert_eq!(
        builder.shapes[1].render_from_object.primary(),
        Transform::identity()
    );
}

#[test]
fn attribute_end_restores_graphics_state_for_following_shapes() {
    let mut builder = SceneBuilder::new();
    let scene = r#"
WorldBegin
AttributeBegin
ReverseOrientation
Shape "sphere"
AttributeEnd
Shape "sphere"
WorldEnd
"#;

    parse_string(scene, &mut builder).expect("scene should parse");
    assert_eq!(builder.shapes.len(), 2);
    assert!(builder.shapes[0].reverse_orientation);
    assert!(!builder.shapes[1].reverse_orientation);
}
