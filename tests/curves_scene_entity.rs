use pbrt_r4::parser::{parse_string, SceneBuilder};

const CURVE: &str = r#"Shape "curve"
    "point3 P" [ 0 0 0  0 0 1  1 0 1  1 0 2 ]"#;

#[test]
fn consecutive_curve_directives_form_one_shape_scene_entity() {
    let mut builder = SceneBuilder::new();
    let scene = format!("WorldBegin\n{CURVE}\n{CURVE}\nWorldEnd\n");

    parse_string(&scene, &mut builder).expect("scene should parse");

    assert_eq!(builder.shapes.len(), 1);
    assert_eq!(builder.shapes[0].base.name, "curves");
    assert!(builder.shapes[0].base.params.get_points_ref("P").is_none());
    assert_eq!(builder.shapes[0].child_params.len(), 2);
}

#[test]
fn non_curve_shape_ends_a_curves_group() {
    let mut builder = SceneBuilder::new();
    let scene = format!("WorldBegin\n{CURVE}\nShape \"sphere\"\n{CURVE}\nWorldEnd\n");

    parse_string(&scene, &mut builder).expect("scene should parse");

    assert_eq!(builder.shapes.len(), 3);
    assert_eq!(builder.shapes[0].base.name, "curves");
    assert_eq!(builder.shapes[1].base.name, "sphere");
    assert_eq!(builder.shapes[2].base.name, "curves");
    assert_eq!(builder.shapes[0].child_params.len(), 1);
    assert!(builder.shapes[1].child_params.is_empty());
    assert_eq!(builder.shapes[2].child_params.len(), 1);
}

#[test]
fn transform_type_and_orientation_changes_end_curves_groups() {
    let mut builder = SceneBuilder::new();
    let scene = format!(
        r#"WorldBegin
{CURVE}
Translate 1 0 0
{CURVE}
Shape "curve" "string type" "cylinder"
    "point3 P" [ 0 0 0  0 0 1  1 0 1  1 0 2 ]
ReverseOrientation
Shape "curve" "string type" "cylinder"
    "point3 P" [ 0 0 0  0 0 1  1 0 1  1 0 2 ]
WorldEnd
"#
    );

    parse_string(&scene, &mut builder).expect("scene should parse");

    assert_eq!(builder.shapes.len(), 4);
    assert!(builder
        .shapes
        .iter()
        .all(|shape| shape.base.name == "curves" && shape.child_params.len() == 1));
}

#[test]
fn area_light_curves_remain_individual_shape_entries() {
    let mut builder = SceneBuilder::new();
    let scene = format!(
        "WorldBegin\nAreaLightSource \"diffuse\" \"rgb L\" [ 1 1 1 ]\n{CURVE}\n{CURVE}\nWorldEnd\n"
    );

    parse_string(&scene, &mut builder).expect("scene should parse");

    assert_eq!(builder.area_lights.len(), 2);
    assert_eq!(builder.shapes.len(), 2);
    assert!(builder
        .shapes
        .iter()
        .all(|shape| shape.base.name == "curve" && shape.child_params.is_empty()));
}

#[test]
fn material_and_alpha_changes_end_curves_groups() {
    let mut builder = SceneBuilder::new();
    let scene = format!(
        r#"WorldBegin
{CURVE}
Material "diffuse"
{CURVE}
Shape "curve"
    "point3 P" [ 0 0 0  0 0 1  1 0 1  1 0 2 ]
    "float alpha" [ 0.5 ]
WorldEnd
"#
    );

    parse_string(&scene, &mut builder).expect("scene should parse");

    assert_eq!(builder.shapes.len(), 3);
    assert!(builder
        .shapes
        .iter()
        .all(|shape| shape.base.name == "curves" && shape.child_params.len() == 1));
}

#[test]
fn twosided_does_not_end_a_curves_group() {
    let mut builder = SceneBuilder::new();
    let scene = format!(
        r#"WorldBegin
{CURVE}
Shape "curve"
    "bool twosided" [ true ]
    "point3 P" [ 0 0 0  0 0 1  1 0 1  1 0 2 ]
WorldEnd
"#
    );

    parse_string(&scene, &mut builder).expect("scene should parse");

    assert_eq!(builder.shapes.len(), 1);
    assert_eq!(builder.shapes[0].child_params.len(), 2);
}

#[test]
fn medium_interface_change_ends_a_curves_group() {
    let mut builder = SceneBuilder::new();
    let scene = format!(
        r#"WorldBegin
{CURVE}
MediumInterface "fog" ""
{CURVE}
WorldEnd
"#
    );

    parse_string(&scene, &mut builder).expect("scene should parse");

    assert_eq!(builder.shapes.len(), 2);
    assert!(builder
        .shapes
        .iter()
        .all(|shape| shape.base.name == "curves" && shape.child_params.len() == 1));
}

#[test]
fn consecutive_animated_curves_form_one_group() {
    let mut builder = SceneBuilder::new();
    let scene = format!(
        r#"WorldBegin
ActiveTransform StartTime
Translate 1 0 0
ActiveTransform EndTime
Translate 2 0 0
ActiveTransform All
{CURVE}
{CURVE}
WorldEnd
"#
    );

    parse_string(&scene, &mut builder).expect("scene should parse");

    assert!(builder.shapes.is_empty());
    assert_eq!(builder.animated_shapes.len(), 1);
    assert_eq!(builder.animated_shapes[0].child_params.len(), 2);
}

#[test]
fn curves_inside_an_object_definition_are_grouped_locally() {
    let mut builder = SceneBuilder::new();
    let scene = format!(
        "WorldBegin\nObjectBegin \"hair\"\n{CURVE}\n{CURVE}\nObjectEnd\nObjectInstance \"hair\"\nWorldEnd\n"
    );

    parse_string(&scene, &mut builder).expect("scene should parse");

    assert!(builder.shapes.is_empty());
    let definition = &builder.instance_definitions["hair"];
    assert_eq!(definition.shapes.len(), 1);
    assert_eq!(definition.shapes[0].child_params.len(), 2);
}
