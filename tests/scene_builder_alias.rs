use pbrt_r4::parser::{parse_string, SceneBuilder};

#[test]
fn material_bumpmap_is_normalized_to_displacement() {
    let mut builder = SceneBuilder::new();
    let scene = r#"
WorldBegin
AttributeBegin
Material "diffuse" "texture bumpmap" [ "disp" ]
AttributeEnd
WorldEnd
"#;

    parse_string(scene, &mut builder).expect("scene should parse");
    assert_eq!(builder.materials.len(), 1);
    let params = &builder.materials[0].base.params;
    assert!(params.get_textures_ref("displacement").is_some());
    assert!(params.get_textures_ref("bumpmap").is_none());
}
