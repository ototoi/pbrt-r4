use std::sync::Arc;
use std::sync::RwLock;

use pbrt_r4::gpu::ir::node::triangle_mesh_from_params;
use pbrt_r4::gpu::ir::node::{
    node_ref_to_json, tessellate_shapes, Camera, CameraComponent, Component, Material,
    MaterialComponent, Node, Shape, ShapeComponent, SphereShape,
};
use pbrt_r4::parser::scene_builder::{
    FileLoc, RenderFromObject, SceneBuilder, SceneEntity, ShapeSceneEntity,
};

#[test]
fn node_components_wrap_declarative_resources() {
    let material = Arc::new(Material {
        name: "shared-diffuse".to_string(),
        kind: "diffuse".to_string(),
        params: Default::default(),
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
fn sphere_is_normalized_to_triangle_mesh_in_node_ir() {
    let child = Arc::new(RwLock::new(Node::new("sphere")));
    child
        .write()
        .unwrap()
        .add_component(Component::Shape(ShapeComponent {
            shape: Shape::Sphere(Box::new(SphereShape {
                params: Default::default(),
            })),
        }));
    let mut root = Node::new("root");
    root.add_child(child);

    tessellate_shapes(&mut root);

    let child = root.children[0].read().unwrap();
    let Component::Shape(ShapeComponent {
        shape: Shape::TriangleMesh(mesh),
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
        }));
    let mut root = Node::new("root");
    root.add_child(child);

    tessellate_shapes(&mut root);

    let child = root.children[0].read().unwrap();
    let Component::Shape(ShapeComponent {
        shape: Shape::TriangleMesh(mesh),
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
