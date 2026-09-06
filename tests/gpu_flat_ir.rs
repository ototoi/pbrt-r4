use std::sync::{Arc, RwLock};

use pbrt_r4::gpu::ir::flat::{flatten_node, validate_scattering_graph, AttributeKind};
use pbrt_r4::gpu::ir::node::{
    complete_triangle_attributes, AreaLight as NodeAreaLight, AreaLightComponent, Camera,
    CameraComponent, Component, Film, FilmComponent, Instance as NodeInstance, InstanceComponent,
    Integrator as NodeIntegrator, IntegratorComponent, Light as NodeLight, LightComponent,
    Material, MaterialComponent, Node, Output, OutputComponent, Sampler as NodeSampler,
    SamplerComponent, Shape, ShapeComponent, Transform, TriangleMeshShape,
};
use pbrt_r4::gpu::ir::node::{Vec2f, Vec3f};
use pbrt_r4::util::spectrum::{Spectrum, SpectrumType};

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
        reverse_orientation: false,
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
    root.add_child(triangle_node("first", "diffuse", [1.0, 2.0, 3.0]));
    root.add_child(triangle_node("second", "dielectric", [4.0, 5.0, 6.0]));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.vertices.len(), 6);
    assert_eq!(scene.indices, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(scene.attribute_tables.spectra.len(), 1);
    assert_eq!(scene.attribute_tables.scalars.len(), 1);
    assert_eq!(scene.materials[0].attributes.len(), 1);
    assert_eq!(scene.materials[1].attributes.len(), 1);
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
    assert_eq!(scene.materials[0].kind, "diffuse");
    assert_eq!(scene.materials[1].kind, "dielectric");
    assert_eq!(scene.materials[0].scattering_model, 0);
    assert_eq!(scene.materials[1].scattering_model, 1);
    assert_eq!(scene.camera.fov, 60.0);
    assert_eq!(scene.camera.screen_window, [-2.0, 2.0, -1.0, 1.0]);
    assert_eq!(scene.viewport.resolution, [64, 32]);
    assert_eq!(scene.resolved_scattering_models.len(), 2);
    assert_eq!(scene.resolved_scattering_models[0].root_kind, "diffuse");
    assert_eq!(scene.resolved_scattering_models[1].root_kind, "dielectric");
    assert!(scene.primitive_distribution_map.offsets == vec![0]);
}

#[test]
fn flatten_node_lowers_area_light_to_instance_and_global_light_handle() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    let area = triangle_node("emitter", "diffuse", [0.0, 0.0, 0.0]);
    {
        let mut node = area.write().unwrap();
        let Component::Shape(shape) = &mut node.components[0] else {
            panic!("expected shape component");
        };
        let Shape::TriangleMesh(mesh) = &mut shape.shape else {
            panic!("expected triangle mesh");
        };
        mesh.positions.push(Vec3f([2.0, 1.0, 0.0]));
        mesh.indices.extend_from_slice(&[1, 3, 2]);
        mesh.normals.as_mut().unwrap().push(Vec3f([0.0, 0.0, 1.0]));
        mesh.uvs.as_mut().unwrap().push(Vec2f([1.0, 1.0]));
        node.add_component(Component::AreaLight(AreaLightComponent {
            area_light: NodeAreaLight {
                name: "diffuse".to_string(),
                params: Default::default(),
            },
        }));
    }
    root.add_child(area);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.instances.len(), 1);
    assert_eq!(scene.instances[0].area_light, 0);
    assert_eq!(scene.area_lights.len(), 1);
    assert_eq!(scene.area_lights[0].instance, 0);
    assert_eq!(scene.area_lights[0].distribution.offset, 0);
    assert_eq!(scene.area_lights[0].distribution.count, 2);
    assert_eq!(scene.area_lights[0].distribution.total_area, 1.5);
    assert_eq!(scene.triangle_distributions.len(), 2);
    assert_eq!(scene.triangle_distributions[0].primitive, 0);
    assert!((scene.triangle_distributions[0].cdf - 1.0 / 3.0).abs() < 1e-6);
    assert_eq!(scene.triangle_distributions[1].primitive, 1);
    assert_eq!(scene.triangle_distributions[1].cdf, 1.0);
    assert_eq!(scene.lights.len(), 1);
    assert_eq!(scene.lights[0].payload, 0);
    assert_eq!(
        scene.lights[0].kind,
        pbrt_r4::gpu::ir::flat::LightKind::Area
    );
    assert_eq!(scene.primitive_distribution_map.offsets, vec![0, 2]);
    assert_eq!(scene.primitive_distribution_map.entries, vec![0, 1]);
}

#[test]
fn flatten_node_rejects_unsupported_area_light_power() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    let area = triangle_node("powered-emitter", "diffuse", [0.0, 0.0, 0.0]);
    let mut params = pbrt_r4::paramdict::ParameterDictionary::default();
    params.add_float("float power", 10.0);
    area.write()
        .unwrap()
        .add_component(Component::AreaLight(AreaLightComponent {
            area_light: NodeAreaLight {
                name: "diffuse".to_string(),
                params,
            },
        }));
    root.add_child(area);

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error.to_string().contains("area light power"));
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

    let target = triangle_node("target", "diffuse", [0.0, 0.0, 0.0]);
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
fn flatten_node_preserves_shape_reverse_orientation() {
    let mut root = Node::new("root");
    let mut camera_params = pbrt_r4::paramdict::ParameterDictionary::default();
    camera_params.add_float("float fov", 60.0);
    add_camera_and_film(&mut root, camera_params);
    let shape = triangle_node("reversed", "diffuse", [0.0, 0.0, 0.0]);
    if let Component::Shape(component) = &mut shape.write().unwrap().components[0] {
        component.reverse_orientation = true;
    }
    root.add_child(shape);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert!(scene.instances[0].reverse_orientation);
}

#[test]
fn flatten_node_requires_tessellated_shapes() {
    let mut root = Node::new("root");
    let mut shape = Node::new("sphere");
    shape.add_component(Component::Shape(ShapeComponent {
        shape: Shape::Sphere(Box::new(pbrt_r4::gpu::ir::node::SphereShape {
            params: Default::default(),
        })),
        reverse_orientation: false,
    }));
    shape.add_component(Component::Material(MaterialComponent {
        material: Arc::new(Material {
            name: "diffuse".to_string(),
            kind: "diffuse".to_string(),
            params: Default::default(),
        }),
    }));
    root.add_child(Arc::new(RwLock::new(shape)));

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(format!("{error:?}").contains("tessellated"));
}

#[test]
fn flatten_node_composes_parent_and_child_transforms() {
    let child = triangle_node("triangle", "diffuse", [0.0, 2.0, 0.0]);
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
    integrator_params.add_string("string lightsampler", "uniform");
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
    assert_eq!(scene.render_settings.light_sampler, "uniform");
    assert_eq!(scene.point_lights.len(), 1);
    assert_eq!(scene.point_lights[0].position, [1.0, 2.0, 3.0]);
}

#[test]
fn flatten_node_defaults_light_sampler_to_bvh() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_component(Component::Integrator(IntegratorComponent {
        integrator: NodeIntegrator {
            name: "path".to_string(),
            params: Default::default(),
        },
    }));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.render_settings.light_sampler, "bvh");
}

#[test]
fn flatten_node_extracts_explicit_diffuse_reflectance() {
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
        .expect("diffuse reflectance should be normalized into Flat IR");
    assert_eq!(scene.materials[0].kind, "diffuse");
    let expected_reflectance = Spectrum::from_rgb(&[0.5, 0.5, 0.5], SpectrumType::Albedo).to_rgb();
    assert_eq!(scene.materials[0].attributes.len(), 1);
    assert_eq!(
        scene.materials[0].attributes[0].kind,
        AttributeKind::Spectrum
    );
    assert_eq!(scene.materials[0].attributes[0].name, "reflectance");
    let attribute = &scene.materials[0].attributes[0];
    let spectrum = scene.attribute_tables.spectra[attribute.index as usize].0;
    assert_eq!(
        spectrum,
        [
            expected_reflectance[0],
            expected_reflectance[1],
            expected_reflectance[2],
            0.0
        ]
    );
    assert_eq!(scene.materials[0].scattering_model, 0);
    assert_eq!(scene.scattering_models[0].surface_root, 0);
    assert_eq!(scene.scattering_nodes[0].kind, "diffuse");
    assert_eq!(scene.scattering_nodes[0].event_flags, 0b00101);
    assert_eq!(scene.scattering_nodes[0].data_index, 0);
    assert_eq!(scene.diffuse_bxdf_data.len(), 1);
}

#[test]
fn flatten_node_extracts_dielectric_eta() {
    let shape = triangle_node("triangle", "dielectric", [0.0, 0.0, 0.0]);
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
            .add_float("float eta", 1.33);
    }
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.materials[0].attributes.len(), 1);
    assert_eq!(scene.materials[0].attributes[0].kind, AttributeKind::Scalar);
    assert_eq!(scene.materials[0].attributes[0].name, "eta");
    let attribute = &scene.materials[0].attributes[0];
    assert_eq!(
        scene.attribute_tables.scalars[attribute.index as usize],
        1.33
    );
    assert_eq!(scene.materials[0].scattering_model, 0);
    assert_eq!(scene.scattering_nodes[0].kind, "dielectric");
    assert_eq!(scene.scattering_nodes[0].event_flags, 0b10011);
    assert_eq!(scene.dielectric_bxdf_data.len(), 1);
}

#[test]
fn flatten_node_extracts_thin_dielectric_leaf() {
    let shape = triangle_node("triangle", "thindielectric", [0.0, 0.0, 0.0]);
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);
    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.materials[0].attributes.len(), 1);
    assert_eq!(scene.materials[0].attributes[0].kind, AttributeKind::Scalar);
    assert_eq!(scene.materials[0].attributes[0].name, "eta");
    assert_eq!(scene.scattering_nodes[0].kind, "thindielectric");
    assert_eq!(scene.scattering_nodes[0].event_flags, 0b10011);
}

#[test]
fn flatten_node_builds_coateddiffuse_layered_graph() {
    let shape = triangle_node("triangle", "coateddiffuse", [0.0, 0.0, 0.0]);
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.materials[0].kind, "coateddiffuse");
    assert_eq!(scene.materials[0].scattering_model, 0);
    assert_eq!(scene.scattering_models[0].surface_root, 2);
    assert_eq!(scene.scattering_nodes.len(), 3);
    assert_eq!(scene.scattering_nodes[0].kind, "dielectric");
    assert_eq!(scene.scattering_nodes[1].kind, "diffuse");
    assert_eq!(scene.scattering_nodes[2].kind, "layered");
    assert_eq!(scene.scattering_nodes[2].child_offset, 0);
    assert_eq!(scene.scattering_nodes[2].child_count, 2);
    assert_eq!(scene.scattering_child_refs.node_ids, vec![0, 1]);
    assert_eq!(scene.layered_bxdf_data[0].thickness, 0.01);
    assert_eq!(scene.layered_bxdf_data[0].max_depth, 10);
    assert_eq!(scene.layered_bxdf_data[0].n_samples, 1);
    assert!(scene.layered_bxdf_data[0].two_sided);
    use pbrt_r4::gpu::ir::flat::{EVENT_DIFFUSE, EVENT_REFLECTION, EVENT_SPECULAR};
    assert_eq!(
        scene.scattering_nodes[2].event_flags,
        EVENT_REFLECTION | EVENT_SPECULAR | EVENT_DIFFUSE
    );
}

#[test]
fn layered_anisotropy_rejects_both_endpoints() {
    for g in [-1.0, 1.0] {
        let shape = triangle_node("layered", "coateddiffuse", [0.0; 3]);
        for component in &mut shape.write().unwrap().components {
            if let Component::Material(component) = component {
                Arc::get_mut(&mut component.material)
                    .unwrap()
                    .params
                    .add_float("float g", g);
            }
        }
        let mut root = Node::new("root");
        add_camera_and_film(&mut root, Default::default());
        root.add_child(shape);
        assert!(flatten_node(Arc::new(RwLock::new(root))).is_err());
    }
}

#[test]
fn material_table_preserves_child_data_indices_in_mixed_scenes() {
    use pbrt_r4::gpu::webgpu::material::MaterialTable;
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    for kind in [
        "diffuse",
        "coateddiffuse",
        "dielectric",
        "coateddiffuse",
        "diffuse",
    ] {
        root.add_child(triangle_node(kind, kind, [0.0; 3]));
    }
    let mut scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    let table = MaterialTable::from_flat(&scene).unwrap();
    assert_eq!(table.diffuse.len(), 4);
    assert_eq!(table.dielectric.len(), 3);
    assert_eq!(table.layered.len(), 2);
    for node in &scene.scattering_nodes {
        let i = node.data_index as usize;
        match node.kind.as_str() {
            "diffuse" => assert!(scene.materials.iter().any(|material| {
                material.attributes.iter().any(|attribute| {
                    attribute.name == "reflectance"
                        && scene.attribute_tables.spectra[attribute.index as usize].0[..3]
                            == table.diffuse[i].reflectance[..3]
                })
            })),
            "dielectric" => assert!(scene.materials.iter().any(|material| {
                material.attributes.iter().any(|attribute| {
                    attribute.name == "eta"
                        && scene.attribute_tables.scalars[attribute.index as usize]
                            == table.dielectric[i].eta
                })
            })),
            "layered" => assert_eq!(table.layered[i].two_sided, 1),
            _ => unreachable!(),
        }
    }
    scene.scattering_nodes[0].data_index = u32::MAX;
    assert!(MaterialTable::from_flat(&scene).is_err());
}

#[test]
#[ignore = "requires a Vulkan GPU with experimental ray queries"]
fn layered_scene_uniform_points_to_layered_table_not_bssrdf_table() {
    use pbrt_r4::gpu::webgpu::{
        abi::INVALID_INDEX, context::Context, scene::Scene, stages::RequiredLimits,
    };
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(triangle_node("layered", "coateddiffuse", [0.0; 3]));
    let flat = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    let context = Context::new(RequiredLimits::default()).unwrap();
    let scene = Scene::from_flat(&context.device, &context.queue, flat).unwrap();
    assert_eq!(scene.material_table.layered_bxdf_count, 1);
    assert_eq!(scene.material_table.bssrdf_node_count, 0);
    assert_eq!(scene.material_table.bssrdf_node_offset_words, INVALID_INDEX);
    assert_eq!(
        scene.material_table.layered_bxdf_offset_words,
        scene.material_table.scattering_child_offset_words
            + scene.material_table.scattering_child_count
    );
    assert_eq!(scene.material_table.dielectric_material_count, 1);
    assert_eq!(scene.material_table.diffuse_material_count, 1);
}

#[test]
fn scattering_graph_validation_rejects_cycles_and_invalid_ranges() {
    let cyclic_nodes = vec![pbrt_r4::gpu::ir::flat::ScatteringNode {
        kind: "layered".to_string(),
        event_flags: 0,
        data_index: 0,
        child_offset: 0,
        child_count: 1,
    }];
    let cyclic_model = vec![pbrt_r4::gpu::ir::flat::ScatteringModel {
        surface_root: 0,
        bssrdf_root: u32::MAX,
    }];
    let error = validate_scattering_graph(
        &cyclic_model,
        &cyclic_nodes,
        &pbrt_r4::gpu::ir::flat::ScatteringChildRefs { node_ids: vec![0] },
    )
    .unwrap_err();
    assert!(error.to_string().contains("Cycle detected"));

    let invalid_range = vec![pbrt_r4::gpu::ir::flat::ScatteringNode {
        kind: "diffuse".to_string(),
        event_flags: 0,
        data_index: 0,
        child_offset: 1,
        child_count: 1,
    }];
    let error = validate_scattering_graph(
        &cyclic_model,
        &invalid_range,
        &pbrt_r4::gpu::ir::flat::ScatteringChildRefs { node_ids: vec![] },
    )
    .unwrap_err();
    assert!(error.to_string().contains("invalid child range"));
}

#[test]
fn flatten_node_rejects_invalid_dielectric_eta() {
    let shape = triangle_node("triangle", "dielectric", [0.0, 0.0, 0.0]);
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
            .add_float("float eta", 0.0);
    }
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error.to_string().contains("invalid dielectric eta"));
}

#[test]
fn flatten_node_rejects_unsupported_material_kind() {
    let shape = triangle_node("triangle", "conductor", [0.0, 0.0, 0.0]);
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error.to_string().contains("Unsupported GPU material kind"));
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
