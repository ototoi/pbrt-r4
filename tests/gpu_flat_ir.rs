use std::sync::{Arc, RwLock};

use pbrt_r4::gpu::ir::flat::flatten_node;
use pbrt_r4::gpu::ir::node::{
    Camera, CameraComponent, Component, Film, FilmComponent, Instance as NodeInstance,
    InstanceComponent, Material, MaterialComponent, Node, Shape, ShapeComponent, Transform,
    TriangleMeshShape,
};
use pbrt_r4::gpu::ir::node::{Vec2f, Vec3f};

fn triangle_node(name: &str, material: &str, offset: [f32; 3]) -> Arc<RwLock<Node>> {
    let mut node = Node::new(name);
    node.transform.matrix[3] = offset[0];
    node.transform.matrix[7] = offset[1];
    node.transform.matrix[11] = offset[2];
    node.add_component(Component::Shape(ShapeComponent {
        shape: Shape::TriangleMesh(Box::new(TriangleMeshShape {
            positions: vec![
                Vec3f([0.0, 0.0, 0.0]),
                Vec3f([1.0, 0.0, 0.0]),
                Vec3f([0.0, 1.0, 0.0]),
            ],
            indices: vec![0, 1, 2],
            normals: Some(vec![Vec3f([0.0, 0.0, 1.0]); 3]),
            tangents: None,
            uvs: Some(vec![
                Vec2f([0.0, 0.0]),
                Vec2f([1.0, 0.0]),
                Vec2f([0.0, 1.0]),
            ]),
        })),
    }));
    node.add_component(Component::Material(MaterialComponent {
        material: Arc::new(Material {
            name: material.to_string(),
            kind: material.to_string(),
            params: Default::default(),
        }),
    }));
    Arc::new(RwLock::new(node))
}

fn add_camera_and_film(root: &mut Node, camera_params: pbrt_r4::paramdict::ParameterDictionary) {
    let mut camera = Node::new("camera");
    camera.add_component(Component::Camera(CameraComponent {
        camera: Camera {
            params: camera_params,
            medium: String::new(),
        },
    }));
    let mut film_params = pbrt_r4::paramdict::ParameterDictionary::default();
    film_params.add_int("integer xresolution", 64);
    film_params.add_int("integer yresolution", 32);
    camera.add_component(Component::Film(FilmComponent {
        film: Film {
            name: "rgb".to_string(),
            params: film_params,
        },
    }));
    root.add_child(Arc::new(RwLock::new(camera)));
}

fn instance_node(name: &str, target: &Arc<RwLock<Node>>, offset: [f32; 3]) -> Arc<RwLock<Node>> {
    let mut node = Node::new(name);
    let mut transform = Transform::default();
    transform.matrix[3] = offset[0];
    transform.matrix[7] = offset[1];
    transform.matrix[11] = offset[2];
    node.add_component(Component::Instance(InstanceComponent {
        instance: NodeInstance {
            target: Arc::clone(target),
            transform,
        },
    }));
    Arc::new(RwLock::new(node))
}

#[test]
fn flatten_node_packs_mesh_ranges_and_instances() {
    let mut root = Node::new("root");
    let mut camera_params = pbrt_r4::paramdict::ParameterDictionary::default();
    camera_params.add_float("float fov", 60.0);
    add_camera_and_film(&mut root, camera_params);
    root.add_child(triangle_node("first", "matte", [1.0, 2.0, 3.0]));
    root.add_child(triangle_node("second", "plastic", [4.0, 5.0, 6.0]));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.vertices.len(), 6);
    assert_eq!(scene.indices, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(
        scene.geometries,
        vec![
            pbrt_r4::gpu::ir::flat::Geometry {
                first_vertex: 0,
                vertex_count: 3,
                first_index: 0,
                index_count: 3,
            },
            pbrt_r4::gpu::ir::flat::Geometry {
                first_vertex: 3,
                vertex_count: 3,
                first_index: 3,
                index_count: 3,
            },
        ]
    );
    assert_eq!(scene.instances.len(), 2);
    assert_eq!(scene.instances[0].geometry, 0);
    assert_eq!(scene.instances[0].material, 0);
    assert_eq!(scene.instances[0].transform[3], 1.0);
    assert_eq!(scene.instances[0].transform[7], 2.0);
    assert_eq!(scene.instances[0].transform[11], 3.0);
    assert_eq!(scene.instances[1].geometry, 1);
    assert_eq!(scene.instances[1].material, 1);
    assert_eq!(scene.materials[0].kind, "matte");
    assert_eq!(scene.materials[1].kind, "plastic");
    assert_eq!(scene.camera.fov, 60.0);
    assert_eq!(scene.camera.screen_window, [-2.0, 2.0, -1.0, 1.0]);
    assert_eq!(scene.viewport.resolution, [64, 32]);
}

#[test]
fn flatten_node_preserves_explicit_camera_screen_window() {
    let mut root = Node::new("root");
    let mut camera = Node::new("camera");
    let mut camera_params = pbrt_r4::paramdict::ParameterDictionary::default();
    camera_params.add_float("float fov", 60.0);
    camera_params.add_float("float halffov", 10.0);
    camera_params.add_float("float frameaspectratio", 1.0);
    camera_params.add_float("float[] screenwindow", -3.0);
    camera_params.add_float("float[] screenwindow", 3.0);
    camera_params.add_float("float[] screenwindow", -2.0);
    camera_params.add_float("float[] screenwindow", 2.0);
    camera.add_component(Component::Camera(CameraComponent {
        camera: Camera {
            params: camera_params,
            medium: String::new(),
        },
    }));
    let mut film_params = pbrt_r4::paramdict::ParameterDictionary::default();
    film_params.add_int("integer xresolution", 64);
    film_params.add_int("integer yresolution", 32);
    camera.add_component(Component::Film(FilmComponent {
        film: Film {
            name: "rgb".to_string(),
            params: film_params,
        },
    }));
    root.add_child(Arc::new(RwLock::new(camera)));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.camera.fov, 60.0);
    assert_eq!(scene.camera.screen_window, [-3.0, 3.0, -2.0, 2.0]);
}

#[test]
fn flatten_node_shares_geometry_across_instances() {
    let mut root = Node::new("root");
    let mut camera_params = pbrt_r4::paramdict::ParameterDictionary::default();
    camera_params.add_float("float fov", 60.0);
    add_camera_and_film(&mut root, camera_params);

    let target = triangle_node("target", "matte", [0.0, 0.0, 0.0]);
    root.add_child(instance_node("first-instance", &target, [1.0, 0.0, 0.0]));
    root.add_child(instance_node("second-instance", &target, [0.0, 2.0, 0.0]));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.vertices.len(), 3);
    assert_eq!(scene.indices, vec![0, 1, 2]);
    assert_eq!(scene.geometries.len(), 1);
    assert_eq!(scene.instances.len(), 2);
    assert_eq!(scene.instances[0].geometry, 0);
    assert_eq!(scene.instances[1].geometry, 0);
    assert_eq!(scene.instances[0].transform[3], 1.0);
    assert_eq!(scene.instances[1].transform[7], 2.0);
}

#[test]
fn flatten_node_requires_tessellated_shapes() {
    let mut root = Node::new("root");
    let mut shape = Node::new("sphere");
    shape.add_component(Component::Shape(ShapeComponent {
        shape: Shape::Sphere(Box::new(pbrt_r4::gpu::ir::node::SphereShape {
            params: Default::default(),
        })),
    }));
    shape.add_component(Component::Material(MaterialComponent {
        material: Arc::new(Material {
            name: "matte".to_string(),
            kind: "matte".to_string(),
            params: Default::default(),
        }),
    }));
    root.add_child(Arc::new(RwLock::new(shape)));

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(format!("{error:?}").contains("tessellated"));
}

#[test]
fn flatten_node_composes_parent_and_child_transforms() {
    let child = triangle_node("triangle", "matte", [0.0, 2.0, 0.0]);
    let mut root = Node::new("root");
    let mut camera_params = pbrt_r4::paramdict::ParameterDictionary::default();
    camera_params.add_float("float fov", 60.0);
    add_camera_and_film(&mut root, camera_params);
    root.transform = Transform {
        matrix: [
            1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0,
        ],
    };
    root.add_child(child);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.instances[0].transform[3], 1.0);
    assert_eq!(scene.instances[0].transform[7], 2.0);
    assert_eq!(scene.instances[0].transform[11], 3.0);
}
