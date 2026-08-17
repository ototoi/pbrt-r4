use pbrt_r4::prelude::*;
use pbrt_r4::shapes::create_ply_mesh;

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

fn write_temp_ply(contents: &str) -> tempfile::NamedTempFile {
    let mut file = tempfile::Builder::new()
        .prefix("pbrt-r4-plymesh-")
        .suffix(".ply")
        .tempfile()
        .expect("temporary ply file should be created");
    file.write_all(contents.as_bytes())
        .expect("ply contents should be written");
    file
}

#[test]
fn plymesh_allows_unknown_vertex_properties() {
    let ply = r#"ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property float foo
element face 1
property list uchar int vertex_indices
end_header
0 0 0 1
1 0 0 1
0 1 0 1
3 0 1 2
"#;
    let file = write_temp_ply(ply);
    let mut params = ParameterDictionary::new();
    params.add_string("filename", file.path().to_string_lossy().as_ref());

    let float_textures: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let shapes = create_ply_mesh(
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
        &float_textures,
    )
    .expect("plymesh with unknown vertex properties should parse");

    assert_eq!(shapes.len(), 1);
}

#[test]
fn plymesh_returns_error_for_invalid_header() {
    let file = write_temp_ply("this is not a valid ply file\n");
    let mut params = ParameterDictionary::new();
    params.add_string("filename", file.path().to_string_lossy().as_ref());

    let float_textures: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let result = create_ply_mesh(
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
        &float_textures,
    );

    assert!(result.is_err());
}

#[test]
fn plymesh_skips_degenerate_triangles() {
    let ply = r#"ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
2 0 0
3 0 1 2
"#;
    let file = write_temp_ply(ply);
    let mut params = ParameterDictionary::new();
    params.add_string("filename", file.path().to_string_lossy().as_ref());

    let float_textures: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let shapes = create_ply_mesh(
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
        &float_textures,
    )
    .expect("degenerate plymesh should be skipped without failing the scene");

    assert!(shapes.is_empty());
}

#[test]
fn plymesh_quads_become_bilinear_patches() {
    let ply = r#"ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
-1 0 -1
1 0 -1
-1 0 1
1 0 1
4 0 1 2 3
"#;
    let file = write_temp_ply(ply);
    let mut params = ParameterDictionary::new();
    params.add_string("filename", file.path().to_string_lossy().as_ref());

    let float_textures: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let shapes = create_ply_mesh(
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
        &float_textures,
    )
    .expect("ply quad should parse into a bilinear patch");

    assert_eq!(shapes.len(), 1);
    assert!(matches!(shapes[0], Shape::BilinearPatch(_)));
}
