use std::sync::{Arc, RwLock};

use pbrt_r4::gpu::ir::flat::flatten_node;
use pbrt_r4::gpu::ir::node::{
    complete_triangle_attributes, AreaLight as NodeAreaLight, AreaLightComponent, Camera,
    CameraComponent, Component, Film, FilmComponent, Instance as NodeInstance, InstanceComponent,
    Integrator as NodeIntegrator, IntegratorComponent, Light as NodeLight, LightComponent,
    Material, MaterialComponent, Node, Output, OutputComponent, Sampler as NodeSampler,
    SamplerComponent, Shape, ShapeComponent, Transform, TriangleMeshShape,
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
    root.add_component(Component::Output(OutputComponent {
        output: Output {
            filename: "test.exr".to_string(),
        },
    }));
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
fn flatten_node_lowers_area_light_to_instance_and_global_light_handle() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    let area = triangle_node("emitter", "diffuse", [0.0, 0.0, 0.0]);
    area.write()
        .unwrap()
        .add_component(Component::AreaLight(AreaLightComponent {
            area_light: NodeAreaLight {
                name: "diffuse".to_string(),
                params: Default::default(),
            },
        }));
    root.add_child(area);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.instances.len(), 1);
    assert_eq!(scene.instances[0].area_light, 0);
    assert_eq!(scene.area_lights.len(), 1);
    assert_eq!(scene.area_lights[0].instance, 0);
    assert_eq!(scene.lights.len(), 1);
    assert_eq!(scene.lights[0].payload, 0);
    assert_eq!(
        scene.lights[0].kind,
        pbrt_r4::gpu::ir::flat::LightKind::Area
    );
}

#[test]
fn flatten_node_preserves_explicit_camera_screen_window() {
    let mut root = Node::new("root");
    root.add_component(Component::Output(OutputComponent {
        output: Output {
            filename: "test.exr".to_string(),
        },
    }));
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

#[test]
fn flatten_node_extracts_render_settings_and_point_lights() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    let mut sampler_params = pbrt_r4::paramdict::ParameterDictionary::default();
    sampler_params.add_int("integer pixelsamples", 8);
    sampler_params.add_int("integer seed", 13);
    root.add_component(Component::Sampler(SamplerComponent {
        sampler: NodeSampler {
            name: "independent".to_string(),
            params: sampler_params,
        },
    }));
    let mut integrator_params = pbrt_r4::paramdict::ParameterDictionary::default();
    integrator_params.add_int("integer maxdepth", 3);
    root.add_component(Component::Integrator(IntegratorComponent {
        integrator: NodeIntegrator {
            name: "path".to_string(),
            params: integrator_params,
        },
    }));
    let mut light_params = pbrt_r4::paramdict::ParameterDictionary::default();
    light_params.add_point("point from", &[1.0, 2.0, 3.0]);
    light_params.add_rgb("rgb I", &[2.0, 2.0, 2.0]);
    let mut light = Node::new("point");
    light.add_component(Component::Light(LightComponent {
        light: NodeLight {
            name: "point".to_string(),
            params: light_params,
            transform: Transform::default(),
            medium: String::new(),
        },
    }));
    root.add_child(Arc::new(RwLock::new(light)));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.render_settings.samples_per_pixel, 8);
    assert_eq!(scene.render_settings.max_depth, 3);
    assert_eq!(scene.render_settings.seed, 13);
    assert_eq!(scene.point_lights.len(), 1);
    assert_eq!(scene.point_lights[0].position, [1.0, 2.0, 3.0]);
}

#[test]
fn flatten_node_ignores_explicit_diffuse_reflectance_for_now() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    let shape = triangle_node("triangle", "diffuse", [0.0, 0.0, 0.0]);
    {
        let mut node = shape.write().unwrap();
        let material = node
            .components
            .iter_mut()
            .find_map(|component| match component {
                Component::Material(component) => Some(&mut component.material),
                _ => None,
            })
            .unwrap();
        Arc::get_mut(material)
            .expect("test material should be uniquely owned")
            .params
            .add_rgb("rgb reflectance", &[0.5, 0.5, 0.5]);
    }
    root.add_child(shape);

    let scene = flatten_node(Arc::new(RwLock::new(root)))
        .expect("reflectance is currently ignored by the GPU material representation");
    assert_eq!(scene.materials[0].kind, "diffuse");
}

#[test]
fn flatten_node_completes_missing_mesh_attributes() {
    let shape = triangle_node("triangle", "diffuse", [0.0, 0.0, 0.0]);
    {
        let mut node = shape.write().unwrap();
        let Component::Shape(shape) = &mut node.components[0] else {
            panic!("expected shape component");
        };
        let Shape::TriangleMesh(mesh) = &mut shape.shape else {
            panic!("expected triangle mesh");
        };
        mesh.normals = None;
        mesh.tangents = None;
        mesh.uvs = None;
    }
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.vertices.len(), 3);
    assert_eq!(scene.vertices[0].normal, [0.0, 0.0, 1.0]);
    assert_eq!(scene.vertices[0].uv, [0.0, 0.0]);
    assert_eq!(scene.vertices[1].uv, [1.0, 0.0]);
    assert_eq!(scene.vertices[2].uv, [0.0, 1.0]);
    assert!(scene.vertices.iter().all(|vertex| {
        vertex.tangent.iter().all(|value| value.is_finite()) && vertex.tangent[0].abs() > 0.9
    }));
}

#[test]
fn missing_normals_expand_shared_vertices_per_triangle() {
    let mesh = TriangleMeshShape {
        positions: vec![
            Vec3f([0.0, 0.0, 0.0]),
            Vec3f([1.0, 0.0, 0.0]),
            Vec3f([1.0, 1.0, 0.0]),
            Vec3f([0.0, 1.0, 1.0]),
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
        normals: None,
        tangents: None,
        uvs: None,
    };

    let completed = complete_triangle_attributes(mesh, "shared").unwrap();
    assert_eq!(completed.positions.len(), 6);
    assert_eq!(completed.indices, vec![0, 1, 2, 3, 4, 5]);
    assert_ne!(
        completed.normals.as_ref().unwrap()[0],
        completed.normals.as_ref().unwrap()[3]
    );
}
