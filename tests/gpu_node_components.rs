use std::sync::Arc;
use std::sync::RwLock;

use pbrt_r4::gpu::ir::flat::flatten_node;
use pbrt_r4::gpu::ir::node::{
    node_ref_to_json, remove_invalid_triangles, tessellate_shapes, triangle_mesh_from_params,
    Camera, CameraComponent, Component, DiskShape, Material, MaterialComponent, Node, Shape,
    ShapeComponent, SphereShape, Texture, TextureComponent, TextureKind, TextureMapping,
    TextureNode, Transform, TriangleMeshShape, Vec3f,
};
use pbrt_r4::parser::parse_string;
use pbrt_r4::parser::scene_builder::{
    FileLoc, RenderFromObject, SceneBuilder, SceneEntity, ShapeSceneEntity,
};

#[test]
fn node_components_wrap_declarative_resources() {
    let material = Arc::new(Material {
        name: "shared-diffuse".to_string(),
        kind: "diffuse".to_string(),
        params: Default::default(),
        material_attributes: Vec::new(),
        texture_attributes: Vec::new(),
    });

    let camera = Component::Camera(CameraComponent {
        camera: Camera {
            params: Default::default(),
            medium: String::new(),
        },
    });
    let mut sphere_params = pbrt_r4::paramdict::ParameterDictionary::default();
    sphere_params.add_float("float radius", 1.0);
    sphere_params.add_int("integer udiv", 8);
    sphere_params.add_int("integer vdiv", 4);
    let shape = Component::Shape(ShapeComponent {
        shape: Shape::Sphere(Box::new(SphereShape {
            params: sphere_params,
        })),
        reverse_orientation: false,
    });
    let material_component = Component::Material(MaterialComponent {
        material: Arc::clone(&material),
    });

    assert!(matches!(camera, Component::Camera(CameraComponent { .. })));
    assert!(matches!(shape, Component::Shape(ShapeComponent { .. })));
    assert!(matches!(
        material_component,
        Component::Material(MaterialComponent { .. })
    ));
    assert_eq!(Arc::strong_count(&material), 2);
}

#[test]
fn texture_node_keeps_mapping_separate_from_texture_data() {
    let mut node = TextureNode::new("albedo");
    node.components.push(TextureComponent::Texture(Texture {
        name: "imagemap".to_string(),
        kind: TextureKind::Spectrum,
        params: Default::default(),
        mipmap: None,
    }));
    node.components
        .push(TextureComponent::Mapping(TextureMapping::PointTransform(
            Transform::default(),
        )));

    assert!(matches!(
        node.components.first(),
        Some(TextureComponent::Texture(Texture { .. }))
    ));
    assert!(matches!(
        node.components.get(1),
        Some(TextureComponent::Mapping(TextureMapping::PointTransform(_)))
    ));
    assert!(node.children.is_empty());
}

#[test]
fn gpu_texture_checkerboard3d_uses_point_transform_mapping() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
Texture "checker" "float" "checkerboard3d"
    "float tex1" [ 0.0 ]
    "float tex2" [ 1.0 ]
"#,
        &mut builder,
    )
    .expect("3D checkerboard texture should parse");
    let root = builder
        .build_gpu_ir_node()
        .expect("GPU node IR should build");
    let flat = flatten_node(root).expect("GPU flat IR should build");
    let checker = flat
        .texture_nodes
        .iter()
        .find(|node| node.name == "checker")
        .expect("checkerboard node should be present");
    assert_eq!(checker.operation, 12);
    assert_eq!(checker.child_count, 2);
}

#[test]
fn gpu_texture_planar_mapping_is_preserved_in_flat_ir() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
Texture "planar" "spectrum" "imagemap"
    "string mapping" [ "planar" ]
    "vector3 v1" [ 0.5 0 0 ]
    "vector3 v2" [ 0 -0.5 0 ]
    "float udelta" [ 0.5 ]
    "float vdelta" [ -0.5 ]
"#,
        &mut builder,
    )
    .expect("planar texture should parse");
    let root = builder
        .build_gpu_ir_node()
        .expect("GPU node IR should build");
    let flat = flatten_node(root).expect("GPU flat IR should build");
    let planar = flat
        .texture_nodes
        .iter()
        .find(|node| node.name == "planar")
        .expect("planar node should be present");
    assert_eq!(planar.mapping_kind, 1);
    assert_eq!(planar.mapping[0], 0.5);
    assert_eq!(planar.mapping[5], -0.5);
    assert_eq!(planar.mapping[3], 0.5);
    assert_eq!(planar.mapping[7], -0.5);
}

#[test]
fn gpu_material_rejects_multiple_texture_names_for_one_parameter() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
Texture "a" "spectrum" "constant" "rgb value" [ 0.1 0.1 0.1 ]
Texture "b" "spectrum" "constant" "rgb value" [ 0.9 0.9 0.9 ]
MakeNamedMaterial "bad" "string type" [ "diffuse" ]
    "texture reflectance" [ "a" "b" ]
ObjectBegin "object"
NamedMaterial "bad"
Shape "sphere"
ObjectEnd
ObjectInstance "object"
"#,
        &mut builder,
    )
    .expect("scene syntax should parse");
    let error = match builder.build_gpu_ir_node() {
        Ok(_) => panic!("multiple texture names should be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("must name exactly one texture"));
}

#[test]
fn gpu_texture_mix_keeps_composite_amount_as_third_child() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
Texture "amount-base" "float" "constant" "float value" [ 0.25 ]
Texture "amount" "float" "scale"
    "texture tex" [ "amount-base" ]
    "float scale" [ 0.5 ]
Texture "mixed" "spectrum" "mix"
    "rgb tex1" [ 0 0 0 ]
    "rgb tex2" [ 1 1 1 ]
    "texture amount" [ "amount" ]
"#,
        &mut builder,
    )
    .expect("composite amount texture should parse");
    let flat = flatten_node(
        builder
            .build_gpu_ir_node()
            .expect("GPU node IR should build"),
    )
    .expect("GPU flat IR should build");
    let mixed = flat
        .texture_nodes
        .iter()
        .find(|node| node.name == "mixed")
        .expect("mixed node should be present");
    assert_eq!(mixed.child_count, 3);
    let amount_index = flat.texture_child_indices[mixed.first_child as usize + 2];
    assert_eq!(flat.texture_nodes[amount_index as usize].name, "amount");
    assert_eq!(flat.texture_nodes[amount_index as usize].operation, 2);
}

#[test]
fn gpu_texture_mix_materializes_literal_operands_as_constant_children() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
Texture "mixed" "spectrum" "mix"
    "rgb tex1" [ 0.2 0.3 0.4 ]
    "rgb tex2" [ 0.8 0.7 0.6 ]
Texture "scaled" "spectrum" "scale"
    "texture tex" [ "mixed" ]
    "float scale" [ 0.5 ]
"#,
        &mut builder,
    )
    .expect("texture definition should parse");

    let root = builder
        .build_gpu_ir_node()
        .expect("GPU node IR should build");
    let flat = flatten_node(Arc::clone(&root)).expect("GPU flat IR should build");
    let scaled = flat
        .texture_nodes
        .iter()
        .find(|node| node.name == "scaled")
        .expect("nested texture node should be present");
    assert_eq!(scaled.child_count, 1);
    let mixed_index = flat.texture_child_indices[scaled.first_child as usize];
    assert_eq!(flat.texture_nodes[mixed_index as usize].name, "mixed");

    let root = root.read().unwrap();
    let scene = root
        .components
        .iter()
        .find_map(|component| match component {
            Component::Scene(component) => Some(&component.scene),
            _ => None,
        });
    let texture = scene
        .and_then(|scene| scene.texture_nodes.first())
        .expect("texture node should be present");
    assert_eq!(texture.children.len(), 2);
    for child in &texture.children {
        assert!(matches!(
            child.components.first(),
            Some(TextureComponent::Texture(Texture { name, .. })) if name == "constant"
        ));
    }
}

#[test]
fn gpu_texture_3d_noise_nodes_use_point_transform_mapping() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
Texture "wrinkles" "float" "wrinkled"
    "integer octaves" [ 4 ]
    "float roughness" [ 0.6 ]
Texture "wind" "float" "windy"
Texture "marble" "spectrum" "marble"
"#,
        &mut builder,
    )
    .expect("3D procedural texture should parse");
    let root = builder
        .build_gpu_ir_node()
        .expect("GPU node IR should build");
    let root_for_flatten = Arc::clone(&root);
    let root = root.read().unwrap();
    let scene = root
        .components
        .iter()
        .find_map(|component| match component {
            Component::Scene(component) => Some(&component.scene),
            _ => None,
        });
    let texture = scene
        .and_then(|scene| scene.texture_nodes.first())
        .expect("texture node should be present");
    assert!(matches!(
        texture.components.get(1),
        Some(TextureComponent::Mapping(TextureMapping::PointTransform(_)))
    ));
    let flat = flatten_node(root_for_flatten).expect("GPU flat IR should build");
    assert_eq!(
        flat.texture_nodes
            .iter()
            .find(|node| node.name == "wrinkles")
            .expect("wrinkled node should be flattened")
            .operation,
        8
    );
    assert_eq!(
        flat.texture_nodes
            .iter()
            .find(|node| node.name == "wind")
            .expect("windy node should be flattened")
            .operation,
        9
    );
    assert_eq!(
        flat.texture_nodes
            .iter()
            .find(|node| node.name == "marble")
            .expect("marble node should be flattened")
            .operation,
        11
    );
}

#[test]
fn gpu_texture_dots_materializes_outside_and_inside_operands() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
Texture "polka" "float" "dots"
    "float outside" [ 0.1 ]
    "float inside" [ 0.9 ]
"#,
        &mut builder,
    )
    .expect("dots texture should parse");
    let root = builder
        .build_gpu_ir_node()
        .expect("GPU node IR should build");
    let root = root.read().unwrap();
    let scene = root
        .components
        .iter()
        .find_map(|component| match component {
            Component::Scene(component) => Some(&component.scene),
            _ => None,
        });
    let texture = scene
        .and_then(|scene| scene.texture_nodes.first())
        .expect("texture node should be present");
    assert_eq!(texture.children.len(), 2);
    assert!(texture.children.iter().all(|child| matches!(
        child.components.first(),
        Some(TextureComponent::Texture(Texture { name, .. })) if name == "constant"
    )));
}

#[test]
fn gpu_texture_bilerp_materializes_four_corner_values() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
Texture "grid" "float" "bilerp"
    "float v00" [ 0.0 ]
    "float v01" [ 1.0 ]
    "float v10" [ 0.25 ]
    "float v11" [ 0.75 ]
"#,
        &mut builder,
    )
    .expect("bilerp texture should parse");
    let root = builder
        .build_gpu_ir_node()
        .expect("GPU node IR should build");
    let root = root.read().unwrap();
    let scene = root
        .components
        .iter()
        .find_map(|component| match component {
            Component::Scene(component) => Some(&component.scene),
            _ => None,
        });
    let texture = scene
        .and_then(|scene| scene.texture_nodes.first())
        .expect("texture node should be present");
    assert_eq!(texture.children.len(), 4);
}

#[test]
fn shared_nodes_remain_mutable_through_their_reference() {
    let child = Arc::new(RwLock::new(Node::new("shared")));
    let mut root = Node::new("root");
    root.add_child(Arc::clone(&child));

    child.write().unwrap().name = "updated".to_string();

    assert_eq!(root.children[0].read().unwrap().name, "updated");
}

#[test]
fn scene_builder_preserves_scene_level_and_camera_parameters() {
    let mut builder = SceneBuilder::new();
    builder.camera_params.add_float("float fov", 45.0);

    let root = builder.build_gpu_ir_node().unwrap();
    let root = root.read().unwrap();
    assert_eq!(root.name, "root");
    assert_eq!(root.components.len(), 6);
    assert_eq!(root.children.len(), 1);

    let camera = root.children[0].read().unwrap();
    assert_eq!(camera.name, "camera");
    assert_eq!(camera.components.len(), 2);
    let has_camera = camera.components.iter().any(|component| match component {
        Component::Camera(camera) => camera.camera.params.get_one_float("fov", 0.0) == 45.0,
        _ => false,
    });
    let has_film = camera
        .components
        .iter()
        .any(|component| matches!(component, Component::Film(_)));
    assert!(has_camera);
    assert!(has_film);
}

#[test]
fn scene_builder_gpu_camera_node_uses_camera_to_world_transform() {
    let mut builder = SceneBuilder::new();
    builder.camera_to_world[0] = pbrt_r4::util::transform::Transform::translate(-1.0, -2.0, -3.0);

    let root = builder.build_gpu_ir_node().unwrap();
    let root = root.read().unwrap();
    let camera = root.children[0].read().unwrap();

    assert_eq!(camera.transform.matrix[3], 1.0);
    assert_eq!(camera.transform.matrix[7], 2.0);
    assert_eq!(camera.transform.matrix[11], 3.0);
}

#[test]
fn disk_is_tessellated_as_a_non_degenerate_triangle_fan() {
    let mut params = pbrt_r4::paramdict::ParameterDictionary::default();
    params.add_float("float radius", 2.0);
    params.add_int("integer udiv", 4);
    params.add_int("integer vdiv", 1);
    let mut root = Node::new("root");
    root.add_component(Component::Shape(ShapeComponent {
        shape: Shape::Disk(Box::new(DiskShape { params })),
        reverse_orientation: false,
    }));

    tessellate_shapes(&mut root).unwrap();

    let Component::Shape(shape) = &root.components[0] else {
        panic!("expected shape component");
    };
    let Shape::TriangleMesh(mesh) = &shape.shape else {
        panic!("expected tessellated disk mesh");
    };
    assert_eq!(mesh.indices.len(), 4 * 3);
    assert_eq!(mesh.positions.len(), 5 + 1);
    assert!(mesh
        .indices
        .chunks_exact(3)
        .all(|triangle| triangle[0] != triangle[1]
            && triangle[1] != triangle[2]
            && triangle[0] != triangle[2]));
    assert!(mesh.normals.is_some());
    assert!(mesh.tangents.is_some());
    assert!(mesh.uvs.is_some());
}

#[test]
fn disk_ring_uses_radial_segments_and_preserves_seam_vertices() {
    let mut params = pbrt_r4::paramdict::ParameterDictionary::default();
    params.add_float("float radius", 2.0);
    params.add_float("float innerradius", 1.0);
    params.add_float("float phimax", 180.0);
    params.add_int("integer udiv", 4);
    params.add_int("integer vdiv", 2);
    let mut root = Node::new("root");
    root.add_component(Component::Shape(ShapeComponent {
        shape: Shape::Disk(Box::new(DiskShape { params })),
        reverse_orientation: false,
    }));

    tessellate_shapes(&mut root).unwrap();

    let Component::Shape(shape) = &root.components[0] else {
        panic!("expected shape component");
    };
    let Shape::TriangleMesh(mesh) = &shape.shape else {
        panic!("expected tessellated disk mesh");
    };
    assert_eq!(mesh.indices.len(), 2 * 4 * 2 * 3);
    assert_eq!(mesh.positions.len(), 3 * 5);
    assert_eq!(mesh.uvs.as_ref().unwrap()[4].0[0], 1.0);
    assert_eq!(mesh.uvs.as_ref().unwrap()[5].0[0], 0.0);
}

#[test]
fn disk_rejects_invalid_parameters_during_tessellation() {
    let mut params = pbrt_r4::paramdict::ParameterDictionary::default();
    params.add_float("float radius", 0.0);
    let mut root = Node::new("root");
    root.add_component(Component::Shape(ShapeComponent {
        shape: Shape::Disk(Box::new(DiskShape { params })),
        reverse_orientation: false,
    }));

    assert!(tessellate_shapes(&mut root).is_err());

    let mut params = pbrt_r4::paramdict::ParameterDictionary::default();
    params.add_float("float phimax", f32::INFINITY);
    let mut root = Node::new("root");
    root.add_component(Component::Shape(ShapeComponent {
        shape: Shape::Disk(Box::new(DiskShape { params })),
        reverse_orientation: false,
    }));
    assert!(tessellate_shapes(&mut root).is_err());
}

#[test]
fn sphere_is_normalized_to_triangle_mesh_in_node_ir() {
    let child = Arc::new(RwLock::new(Node::new("sphere")));
    child
        .write()
        .unwrap()
        .add_component(Component::Shape(ShapeComponent {
            shape: Shape::Sphere(Box::new(SphereShape {
                params: Default::default(),
            })),
            reverse_orientation: false,
        }));
    let mut root = Node::new("root");
    root.add_child(child);

    tessellate_shapes(&mut root).unwrap();

    let child = root.children[0].read().unwrap();
    let Component::Shape(ShapeComponent {
        shape: Shape::TriangleMesh(mesh),
        ..
    }) = &child.components[0]
    else {
        panic!("sphere was not tessellated to a triangle mesh");
    };
    let tangents = mesh.tangents.as_ref().expect("sphere tangents");
    assert_eq!(tangents.len(), mesh.positions.len());
    assert!(tangents.iter().all(|tangent| {
        tangent.0.iter().all(|value| value.is_finite()) && tangent.0[0].hypot(tangent.0[1]) > 0.0
    }));
}

#[test]
fn sphere_tessellation_does_not_emit_degenerate_triangles() {
    let child = Arc::new(RwLock::new(Node::new("sphere")));
    child
        .write()
        .unwrap()
        .add_component(Component::Shape(ShapeComponent {
            shape: Shape::Sphere(Box::new(SphereShape {
                params: Default::default(),
            })),
            reverse_orientation: false,
        }));
    let mut root = Node::new("root");
    root.add_child(child);

    tessellate_shapes(&mut root).unwrap();

    let child = root.children[0].read().unwrap();
    let Component::Shape(ShapeComponent {
        shape: Shape::TriangleMesh(mesh),
        ..
    }) = &child.components[0]
    else {
        panic!("sphere was not tessellated to a triangle mesh");
    };
    assert!(mesh.indices.chunks_exact(3).all(|triangle| {
        let p0 = mesh.positions[triangle[0] as usize].0;
        let p1 = mesh.positions[triangle[1] as usize].0;
        let p2 = mesh.positions[triangle[2] as usize].0;
        let edge0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let edge1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cross = [
            edge0[1] * edge1[2] - edge0[2] * edge1[1],
            edge0[2] * edge1[0] - edge0[0] * edge1[2],
            edge0[0] * edge1[1] - edge0[1] * edge1[0],
        ];
        cross.iter().map(|value| value * value).sum::<f32>() > 0.0
    }));
}

#[test]
fn invalid_triangles_are_removed_before_attribute_completion() {
    let shape = TriangleMeshShape {
        positions: vec![
            Vec3f([0.0, 0.0, 0.0]),
            Vec3f([1.0, 0.0, 0.0]),
            Vec3f([0.0, 1.0, 0.0]),
        ],
        indices: vec![0, 1, 2, 0, 1, 1],
        normals: None,
        tangents: None,
        uvs: None,
    };

    let filtered = remove_invalid_triangles(shape).unwrap();
    assert_eq!(filtered.indices, vec![0, 1, 2]);
}

#[test]
fn an_entirely_invalid_triangle_mesh_is_removed() {
    let shape = TriangleMeshShape {
        positions: vec![Vec3f([0.0, 0.0, 0.0]), Vec3f([1.0, 0.0, 0.0])],
        indices: vec![0, 1, 1],
        normals: None,
        tangents: None,
        uvs: None,
    };

    assert!(remove_invalid_triangles(shape).unwrap().indices.is_empty());
}

#[test]
fn plymesh_is_realized_as_triangle_mesh_in_node_ir() {
    let filename = format!("pbrt-r4-gpu-node-{}-{}.ply", std::process::id(), "triangle");
    let directory = std::env::temp_dir();
    let path = directory.join(&filename);
    let ply = "ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float u
property float v
element face 1
property list uchar int vertex_indices
end_header
0 0 0 0 0 1 0 0
1 0 0 0 0 1 1 0
0 1 0 0 0 1 0 1
3 0 1 2
";
    std::fs::write(&path, ply).unwrap();

    let mut params = pbrt_r4::paramdict::ParameterDictionary::default();
    params.add_string("string filename", &filename);
    let mut builder = SceneBuilder::new();
    builder
        .seen_work_dirs
        .push(directory.to_string_lossy().into_owned());
    builder.shapes.push(ShapeSceneEntity {
        base: SceneEntity::new("plymesh", params, FileLoc::default()),
        child_params: Vec::new(),
        render_from_object: RenderFromObject::default(),
        reverse_orientation: false,
        material_index: usize::MAX,
        material_name: None,
        area_light_index: None,
        material_is_default: false,
        medium_interface: Default::default(),
        instance_name: None,
    });

    let root = builder.build_gpu_ir_node().unwrap();
    std::fs::remove_file(path).unwrap();
    let root = root.read().unwrap();
    let shape_node = root.children.iter().find_map(|child| {
        let child = child.read().unwrap();
        child
            .components
            .iter()
            .find_map(|component| match component {
                Component::Shape(ShapeComponent {
                    shape: Shape::TriangleMesh(mesh),
                    ..
                }) => Some((
                    mesh.positions.len(),
                    mesh.indices.len(),
                    mesh.normals.is_some(),
                    mesh.uvs.is_some(),
                )),
                _ => None,
            })
    });
    assert_eq!(shape_node, Some((3, 3, true, true)));
}

#[test]
fn malformed_mesh_attribute_is_rejected_in_node_ir() {
    let mut params = pbrt_r4::paramdict::ParameterDictionary::default();
    params.add_point("P", &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    params.add_int("indices", 0);
    params.add_int("indices", 1);
    params.add_int("indices", 2);
    params.add_point("N", &[0.0, 0.0, 1.0]);

    let error = triangle_mesh_from_params("trianglemesh", &params).unwrap_err();
    assert!(error.to_string().contains("attribute \"N\""));
}

#[test]
fn node_ir_can_be_inspected_as_json() {
    let mut builder = SceneBuilder::new();
    builder.camera_params.add_float("float fov", 45.0);

    let root = builder.build_gpu_ir_node().unwrap();
    let json = node_ref_to_json(&root);

    assert_eq!(json["name"], "root");
    assert_eq!(json["transform"].as_array().unwrap().len(), 16);
    assert_eq!(json["components"][0]["type"], "Scene");
    assert_eq!(json["children"][0]["name"], "camera");
    assert_eq!(
        json["children"][0]["components"][0]["params"]["fov"]["values"][0],
        45.0
    );
}
