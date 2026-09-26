use std::f32::consts::PI;
use std::sync::{Arc, RwLock};

use image::{ImageBuffer, Rgb};
use pbrt_r4::gpu::flat::portal::prepare_portal_image;
use pbrt_r4::gpu::flat::texture::{build_linear_rgb_mipmap, ColorSpace};
use pbrt_r4::gpu::flat::{
    area_light_flags, area_light_is_two_sided, area_light_is_zero_alpha_sample_only,
    evaluate_dense_spectrum, flatten_node, validate_dense_spectra, AttributeKind, SamplerKind,
    SamplerRandomization, Scene as FlatScene,
};
use pbrt_r4::gpu::node::{
    complete_triangle_attributes, prepare_triangle_meshes, tessellate_shapes,
    AreaLight as NodeAreaLight, AreaLightComponent, Camera, CameraComponent, Component, Film,
    FilmComponent, Instance as NodeInstance, InstanceComponent, Integrator as NodeIntegrator,
    IntegratorComponent, Light as NodeLight, LightComponent, Material, MaterialComponent, Node,
    Output, OutputComponent, Sampler as NodeSampler, SamplerComponent, Shape, ShapeComponent,
    Transform, TriangleMeshShape,
};
use pbrt_r4::gpu::node::{Vec2f, Vec3f};
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::util::spectrum::rgb_to_spectrum::{ACES2065_1, SRGB};
use pbrt_r4::util::spectrum::{spectrum_to_photometric, Spectrum, SpectrumType};

#[test]
fn area_light_flags_use_named_encoders_and_decoders() {
    let flags = area_light_flags(true, true);
    assert!(area_light_is_two_sided(flags));
    assert!(area_light_is_zero_alpha_sample_only(flags));

    let flags = area_light_flags(false, false);
    assert!(!area_light_is_two_sided(flags));
    assert!(!area_light_is_zero_alpha_sample_only(flags));
}

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
            tangents: Some(vec![Vec3f([1.0, 0.0, 0.0]); 3]),
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
            material_attributes: Vec::new(),
            texture_attributes: Vec::new(),
        }),
    }));
    Arc::new(RwLock::new(node))
}

fn add_camera_and_film(root: &mut Node, camera_params: ParameterDictionary) {
    add_camera_and_named_film(root, camera_params, "rgb");
}

fn add_camera_and_named_film(root: &mut Node, camera_params: ParameterDictionary, film_name: &str) {
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
    let mut film_params = ParameterDictionary::default();
    film_params.add_int("integer xresolution", 64);
    film_params.add_int("integer yresolution", 32);
    camera.add_component(Component::Film(FilmComponent {
        film: Film {
            name: film_name.to_string(),
            params: film_params,
        },
    }));
    root.add_child(Arc::new(RwLock::new(camera)));
}

#[test]
fn flatten_node_treats_gbuffer_film_as_rgb_output() {
    let mut root = Node::new("root");
    add_camera_and_named_film(&mut root, Default::default(), "gbuffer");

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.viewport.resolution, [64, 32]);
    assert_eq!(scene.film.sensor_response, [0, 1, 2]);
}

#[test]
fn flatten_node_still_rejects_other_film_types() {
    let mut root = Node::new("root");
    add_camera_and_named_film(&mut root, Default::default(), "spectral");

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();

    assert!(error
        .to_string()
        .contains("does not support film \"spectral\""));
}

fn light_node(name: &str, light_kind: &str, params: ParameterDictionary) -> Arc<RwLock<Node>> {
    let mut node = Node::new(name);
    node.add_component(Component::Light(LightComponent {
        light: NodeLight {
            name: light_kind.to_string(),
            params,
            transform: Transform::default(),
            medium: String::new(),
        },
    }));
    Arc::new(RwLock::new(node))
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
    let mut camera_params = ParameterDictionary::default();
    camera_params.add_float("float fov", 60.0);
    add_camera_and_film(&mut root, camera_params);
    root.add_child(triangle_node("first", "diffuse", [1.0, 2.0, 3.0]));
    root.add_child(triangle_node("second", "dielectric", [4.0, 5.0, 6.0]));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.vertices.len(), 6);
    assert_eq!(scene.indices, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(scene.spectrum_attributes.len(), 5);
    assert_eq!(scene.film.sensor_response, [0, 1, 2]);
    assert_eq!(scene.film.imaging_ratio, 1.0);
    validate_dense_spectra(&scene.spectrum_attributes).unwrap();
    assert_eq!(scene.scalar_attributes.len(), 3);
    assert_eq!(scene.material_nodes[0].attributes.len(), 1);
    assert_eq!(scene.material_nodes[1].attributes.len(), 4);
    assert_eq!(
        scene.geometries,
        vec![
            pbrt_r4::gpu::flat::Geometry {
                first_vertex: 0,
                vertex_count: 3,
                first_index: 0,
                index_count: 3,
            },
            pbrt_r4::gpu::flat::Geometry {
                first_vertex: 3,
                vertex_count: 3,
                first_index: 3,
                index_count: 3,
            },
        ]
    );
    assert_eq!(scene.instances.len(), 2);
    assert_eq!(scene.instances[0].geometry, 0);
    assert_eq!(scene.instances[0].material_root, 0);
    assert_eq!(scene.instances[0].transform[3], 1.0);
    assert_eq!(scene.instances[0].transform[7], 2.0);
    assert_eq!(scene.instances[0].transform[11], 3.0);
    assert_eq!(scene.instances[1].geometry, 1);
    assert_eq!(scene.instances[1].material_root, 1);
    assert_eq!(scene.material_nodes[0].kind, "diffuse");
    assert_eq!(scene.material_nodes[1].kind, "dielectric");
    assert_eq!(scene.camera.fov, 60.0);
    assert_eq!(scene.camera.screen_window, [-2.0, 2.0, -1.0, 1.0]);
    assert_eq!(scene.viewport.resolution, [64, 32]);
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
        mesh.tangents.as_mut().unwrap().push(Vec3f([1.0, 0.0, 0.0]));
        mesh.uvs.as_mut().unwrap().push(Vec2f([1.0, 1.0]));
        node.add_component(Component::AreaLight(AreaLightComponent {
            area_light: NodeAreaLight {
                name: "diffuse".to_string(),
                params: Default::default(),
            },
        }));
    }
    root.add_child(area);
    prepare_triangle_meshes(&mut root).unwrap();

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.instances.len(), 1);
    assert_eq!(scene.instances[0].area_light, 0);
    assert_eq!(scene.light_sampling_models.len(), 1);
    assert_eq!(scene.light_sampling_models[0].geometry_index, 0);
    assert_eq!(scene.light_sampling_models[0].distribution_offset, 0);
    assert_eq!(scene.light_sampling_models[0].distribution_count, 2);
    assert_eq!(scene.light_sampling_models[0].total_area, 1.5);
    assert_eq!(scene.triangle_distributions.len(), 2);
    assert_eq!(scene.triangle_distributions[0].primitive, 0);
    assert!((scene.triangle_distributions[0].cdf - 1.0 / 3.0).abs() < 1e-6);
    assert_eq!(scene.triangle_distributions[1].primitive, 1);
    assert_eq!(scene.triangle_distributions[1].cdf, 1.0);
    assert_eq!(scene.lights.len(), 1);
    assert_eq!(scene.lights[0].sampling_model, 0);
    assert_eq!(scene.lights[0].kind, pbrt_r4::gpu::flat::LightKind::Area);
    assert_eq!(scene.primitive_distribution_map.offsets, vec![0, 2]);
    assert_eq!(scene.primitive_distribution_map.entries, vec![0, 1]);
    let scale = &scene.lights[0].attributes[1];
    let expected = 1.0 / spectrum_to_photometric(&Spectrum::from(1.0));
    assert!((scene.scalar_attributes[scale.index as usize] - expected).abs() < 1e-6);
}

#[test]
fn flatten_node_rejects_zero_tangents() {
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
        mesh.tangents = Some(vec![Vec3f([0.0, 0.0, 0.0]); 3]);
    }
    root.add_child(area);

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error.to_string().contains("zero or non-finite tangent"));
}

#[test]
fn flatten_node_rejects_tangents_parallel_to_normal() {
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
        mesh.tangents = Some(vec![Vec3f([0.0, 0.0, 1.0]); 3]);
    }
    root.add_child(area);

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error
        .to_string()
        .contains("not reasonably orthogonal to its normal"));
}

#[test]
fn flatten_node_rejects_unsupported_area_light_power() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    let area = triangle_node("powered-emitter", "diffuse", [0.0, 0.0, 0.0]);
    let mut params = ParameterDictionary::default();
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
    let mut camera_params = ParameterDictionary::default();
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
    let mut film_params = ParameterDictionary::default();
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
    let mut camera_params = ParameterDictionary::default();
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
    assert_eq!(scene.material_roots.len(), 1);
    assert_eq!(scene.material_nodes.len(), 1);
    assert_eq!(
        scene.instances[0].material_root,
        scene.instances[1].material_root
    );
    assert_eq!(scene.instances[0].transform[3], 1.0);
    assert_eq!(scene.instances[1].transform[7], 2.0);
}

#[test]
fn flatten_node_preserves_shape_reverse_orientation() {
    let mut root = Node::new("root");
    let mut camera_params = ParameterDictionary::default();
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
        shape: Shape::Sphere(Box::new(pbrt_r4::gpu::node::SphereShape {
            params: Default::default(),
        })),
        reverse_orientation: false,
    }));
    shape.add_component(Component::Material(MaterialComponent {
        material: Arc::new(Material {
            name: "diffuse".to_string(),
            kind: "diffuse".to_string(),
            params: Default::default(),
            material_attributes: Vec::new(),
            texture_attributes: Vec::new(),
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
    let mut camera_params = ParameterDictionary::default();
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
    let mut sampler_params = ParameterDictionary::default();
    sampler_params.add_int("integer pixelsamples", 8);
    sampler_params.add_int("integer seed", 13);
    root.add_component(Component::Sampler(SamplerComponent {
        sampler: NodeSampler {
            name: "independent".to_string(),
            params: sampler_params,
        },
    }));
    let mut integrator_params = ParameterDictionary::default();
    integrator_params.add_int("integer maxdepth", 3);
    integrator_params.add_string("string lightsampler", "uniform");
    root.add_component(Component::Integrator(IntegratorComponent {
        integrator: NodeIntegrator {
            name: "path".to_string(),
            params: integrator_params,
        },
    }));
    let mut light_params = ParameterDictionary::default();
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
    assert_eq!(scene.render_settings.sampler_kind, SamplerKind::Independent);
    assert_eq!(
        scene.render_settings.randomization,
        SamplerRandomization::None
    );
    assert_eq!(scene.render_settings.max_depth, 3);
    assert_eq!(scene.render_settings.seed, 13);
    assert_eq!(scene.render_settings.light_sampler, "uniform");
    assert_eq!(scene.light_positions, vec![[1.0, 2.0, 3.0]]);
    assert_eq!(scene.lights[0].attributes.len(), 2);
    let scale = &scene.lights[0].attributes[1];
    let intensity = Spectrum::from_rgb(&[2.0, 2.0, 2.0], SpectrumType::Illuminant);
    let expected = 1.0 / spectrum_to_photometric(&intensity);
    assert!((scene.scalar_attributes[scale.index as usize] - expected).abs() < 1e-6);
}

#[test]
fn flatten_node_preserves_halton_sampler_settings() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    let mut params = ParameterDictionary::default();
    params.add_int("integer seed", 29);
    params.add_string("string randomization", "none");
    root.add_component(Component::Sampler(SamplerComponent {
        sampler: NodeSampler {
            name: "halton".to_string(),
            params,
        },
    }));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.render_settings.sampler_kind, SamplerKind::Halton);
    assert_eq!(
        scene.render_settings.randomization,
        SamplerRandomization::None
    );
    assert_eq!(scene.render_settings.samples_per_pixel, 16);
    assert_eq!(scene.render_settings.seed, 29);
}

#[test]
fn flatten_node_uses_halton_defaults() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_component(Component::Sampler(SamplerComponent {
        sampler: NodeSampler {
            name: "halton".to_string(),
            params: Default::default(),
        },
    }));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.render_settings.samples_per_pixel, 16);
    assert_eq!(
        scene.render_settings.randomization,
        SamplerRandomization::PermuteDigits
    );
}

#[test]
fn flatten_node_preserves_remaining_sampler_settings() {
    let cases = [
        ("sobol", SamplerKind::Sobol, SamplerRandomization::FastOwen),
        (
            "paddedsobol",
            SamplerKind::PaddedSobol,
            SamplerRandomization::FastOwen,
        ),
        (
            "zsobol",
            SamplerKind::ZSobol,
            SamplerRandomization::FastOwen,
        ),
        ("pmj02bn", SamplerKind::Pmj02Bn, SamplerRandomization::None),
        (
            "stratified",
            SamplerKind::Stratified,
            SamplerRandomization::None,
        ),
    ];
    for (name, kind, randomization) in cases {
        let mut root = Node::new("root");
        add_camera_and_film(&mut root, Default::default());
        root.add_component(Component::Sampler(SamplerComponent {
            sampler: NodeSampler {
                name: name.to_string(),
                params: Default::default(),
            },
        }));
        let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
        assert_eq!(scene.render_settings.sampler_kind, kind);
        assert_eq!(scene.render_settings.randomization, randomization);
        assert_eq!(scene.render_settings.samples_per_pixel, 16);
    }
}

#[test]
fn flatten_node_normalizes_zsobol_samples_to_power_of_two() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    let mut params = ParameterDictionary::default();
    params.add_int("integer pixelsamples", 13);
    root.add_component(Component::Sampler(SamplerComponent {
        sampler: NodeSampler {
            name: "zsobol".to_string(),
            params,
        },
    }));
    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.render_settings.samples_per_pixel, 8);
}

#[test]
fn flatten_node_rejects_unimplemented_gpu_sampler() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_component(Component::Sampler(SamplerComponent {
        sampler: NodeSampler {
            name: "random".to_string(),
            params: Default::default(),
        },
    }));

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(format!("{error:?}").contains("not implemented"));
}

#[test]
fn flatten_node_rejects_unimplemented_halton_randomization() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    let mut params = ParameterDictionary::default();
    params.add_string("string randomization", "fastowen");
    root.add_component(Component::Sampler(SamplerComponent {
        sampler: NodeSampler {
            name: "halton".to_string(),
            params,
        },
    }));

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(format!("{error:?}").contains("randomization 'fastowen' is not supported"));
}

#[test]
fn flatten_node_separates_spot_and_distant_light_sampling_models() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());

    let mut distant_params = ParameterDictionary::default();
    distant_params.add_point("point from", &[0.0, 0.0, 2.0]);
    distant_params.add_point("point to", &[0.0, 0.0, 0.0]);
    root.add_child(light_node("distant", "distant", distant_params));

    let mut spot_params = ParameterDictionary::default();
    spot_params.add_point("point from", &[1.0, 2.0, 3.0]);
    spot_params.add_point("point to", &[1.0, 2.0, 4.0]);
    spot_params.add_float("float coneangle", 40.0);
    spot_params.add_float("float conedelta", 10.0);
    spot_params.add_float("float power", 10.0);
    let spot = light_node("spot", "spot", spot_params);
    spot.write().unwrap().transform.matrix = [
        2.0, 0.5, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    root.add_child(spot);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    assert_eq!(scene.lights.len(), 1);
    assert_eq!(scene.infinite_lights.len(), 1);
    assert_eq!(scene.lights[0].kind, pbrt_r4::gpu::flat::LightKind::Spot);
    assert_eq!(scene.lights[0].attributes.len(), 4);
    assert_eq!(
        scene.infinite_lights[0].kind,
        pbrt_r4::gpu::flat::LightKind::Distant
    );
    assert_eq!(scene.light_bvh.bounded_handles, vec![0]);

    let spot_model = &scene.light_sampling_models[scene.lights[0].sampling_model as usize];
    assert_eq!(spot_model.kind, pbrt_r4::gpu::flat::LightKind::Spot);
    assert_eq!(
        spot_model.geometry_kind,
        pbrt_r4::gpu::flat::LightGeometryKind::Position
    );
    let distant_model =
        &scene.light_sampling_models[scene.infinite_lights[0].sampling_model as usize];
    assert_eq!(distant_model.kind, pbrt_r4::gpu::flat::LightKind::Distant);
    assert_eq!(
        distant_model.geometry_kind,
        pbrt_r4::gpu::flat::LightGeometryKind::Direction
    );
    assert_ne!(scene.lights[0].sampling_model, 0);
    assert_eq!(scene.infinite_lights[0].sampling_model, 0);
    assert_eq!(
        spot_model.world_to_light,
        [
            [0.5, -1.0 / 12.0, 0.0, 0.0],
            [0.0, 1.0 / 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.0],
        ]
    );

    let cos_start = scene.scalar_attributes[scene.lights[0].attributes[2].index as usize];
    let cos_end = scene.scalar_attributes[scene.lights[0].attributes[3].index as usize];
    assert!((cos_start - 30.0_f32.to_radians().cos()).abs() < 1e-6);
    assert!((cos_end - 40.0_f32.to_radians().cos()).abs() < 1e-6);
    let scale = scene.scalar_attributes[scene.lights[0].attributes[1].index as usize];
    let intensity = Spectrum::from(1.0);
    let k_e = 2.0 * PI * ((1.0 - cos_start) + (cos_start - cos_end) / 2.0);
    let expected_scale = 10.0 / (spectrum_to_photometric(&intensity) * k_e);
    assert!((scale - expected_scale).abs() < 1e-6);
}

#[test]
fn flatten_node_classifies_infinite_light_variants() {
    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("environment.png");
    ImageBuffer::<Rgb<u8>, _>::from_raw(2, 2, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        .unwrap()
        .save(&image_path)
        .unwrap();
    let cases = [
        (
            Default::default(),
            pbrt_r4::gpu::flat::LightKind::UniformInfinite,
        ),
        (
            {
                let mut params = ParameterDictionary::default();
                params.add_string("string filename", image_path.to_str().unwrap());
                params.add_string("string encoding", "linear");
                params
            },
            pbrt_r4::gpu::flat::LightKind::ImageInfinite,
        ),
    ];
    for (params, expected_kind) in cases {
        let mut root = Node::new("root");
        add_camera_and_film(&mut root, Default::default());
        root.add_child(light_node("infinite", "infinite", params));
        let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
        assert_eq!(scene.infinite_lights[0].kind, expected_kind);
        assert_eq!(scene.light_sampling_models.len(), 1);
        assert_eq!(
            scene.light_sampling_models[0].geometry_kind,
            pbrt_r4::gpu::flat::LightGeometryKind::Direction
        );
    }
}

#[test]
fn flatten_node_accepts_portal_infinite_image() {
    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("environment.png");
    ImageBuffer::<Rgb<u8>, _>::from_raw(2, 2, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        .unwrap()
        .save(&image_path)
        .unwrap();
    let mut params = ParameterDictionary::default();
    params.add_string("string filename", image_path.to_str().unwrap());
    params.add_point(
        "point portal",
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    );
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("portal", "infinite", params));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(
        scene.infinite_lights[0].kind,
        pbrt_r4::gpu::flat::LightKind::PortalImageInfinite
    );
    assert_eq!(
        scene.light_sampling_models[0].geometry_kind,
        pbrt_r4::gpu::flat::LightGeometryKind::Portal
    );
    assert_eq!(scene.light_sampling_models[0].distribution_offset, 0);
    assert_eq!(scene.light_sampling_models[0].distribution_count, 0);
    assert_eq!(scene.light_sampling_models[0].total_area, 0.0);
}

#[test]
fn flatten_node_keeps_multiple_portal_images_and_distributions_separate() {
    let directory = tempfile::tempdir().unwrap();
    let red_path = directory.path().join("red.png");
    let blue_path = directory.path().join("blue.png");
    ImageBuffer::<Rgb<u8>, _>::from_pixel(3, 3, Rgb([255, 0, 0]))
        .save(&red_path)
        .unwrap();
    ImageBuffer::<Rgb<u8>, _>::from_pixel(3, 3, Rgb([0, 0, 255]))
        .save(&blue_path)
        .unwrap();

    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    for (name, image_path, x) in [("red", red_path, 0.0), ("blue", blue_path, 2.0)] {
        let mut params = ParameterDictionary::default();
        params.add_string("string filename", image_path.to_str().unwrap());
        params.add_string("string encoding", "linear");
        params.add_point(
            "point portal",
            &[
                x,
                0.0,
                1.0,
                x,
                1.0,
                1.0,
                x + 1.0,
                1.0,
                1.0,
                x + 1.0,
                0.0,
                1.0,
            ],
        );
        root.add_child(light_node(name, "infinite", params));
    }

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.infinite_lights.len(), 2);
    assert_eq!(scene.portal_infinite_lights.len(), 2);
    assert_eq!(scene.light_sampling_models[0].geometry_index, 0);
    assert_eq!(scene.light_sampling_models[1].geometry_index, 1);
    assert_eq!(scene.portal_infinite_lights[0].distribution_offset, 0);
    assert_eq!(scene.portal_infinite_lights[1].distribution_offset, 9);
    assert_eq!(scene.portal_distribution.len(), 18);
    assert_ne!(
        scene.infinite_lights[0].image_index,
        scene.infinite_lights[1].image_index
    );
    scene.validate_static_views().unwrap();
}

fn one_pixel_portal_scene() -> FlatScene {
    let mut params = ParameterDictionary::default();
    params.add_rgb("rgb L", &[1.0, 1.0, 1.0]);
    params.add_point(
        "point portal",
        &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    );
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("portal", "infinite", params));
    flatten_node(Arc::new(RwLock::new(root))).unwrap()
}

#[test]
fn portal_static_view_validation_rejects_inconsistent_flat_data() {
    let scene = one_pixel_portal_scene();

    let mut missing_record = scene.clone();
    missing_record.portal_infinite_lights.clear();
    assert!(missing_record
        .validate_static_views()
        .unwrap_err()
        .to_string()
        .contains("not one-to-one"));

    let mut legacy_fields = scene.clone();
    legacy_fields.light_sampling_models[0].distribution_count = 1;
    assert!(legacy_fields
        .validate_static_views()
        .unwrap_err()
        .to_string()
        .contains("invalid legacy geometry fields"));

    let mut non_finite_geometry = scene.clone();
    non_finite_geometry.portal_infinite_lights[0].portal[0][0] = f32::NAN;
    assert!(non_finite_geometry
        .validate_static_views()
        .unwrap_err()
        .to_string()
        .contains("non-finite geometry"));

    let mut non_finite_distribution = scene.clone();
    non_finite_distribution.portal_distribution[0].summed_area = f32::INFINITY;
    assert!(non_finite_distribution
        .validate_static_views()
        .unwrap_err()
        .to_string()
        .contains("non-finite value"));

    let mut image_mismatch = scene.clone();
    let image = image_mismatch.infinite_lights[0].image_index as usize;
    image_mismatch.texture_library.mipmaps[image] = Arc::new({
        let mut mipmap = (*image_mismatch.texture_library.mipmaps[image]).clone();
        mipmap.levels[0].resolution = [2, 1];
        mipmap
    });
    assert!(image_mismatch
        .validate_static_views()
        .unwrap_err()
        .to_string()
        .contains("resolutions differ"));

    let mut orphan_texel = scene;
    orphan_texel
        .portal_distribution
        .push(orphan_texel.portal_distribution[0]);
    assert!(orphan_texel
        .validate_static_views()
        .unwrap_err()
        .to_string()
        .contains("not owned"));
}

#[test]
fn flatten_node_rejects_portal_without_an_image_source() {
    let mut params = ParameterDictionary::default();
    params.add_point(
        "point portal",
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    );
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("portal", "infinite", params));

    assert!(flatten_node(Arc::new(RwLock::new(root))).is_err());
}

#[test]
fn flatten_node_rejects_non_finite_final_portal_scale() {
    let mut params = ParameterDictionary::default();
    params.add_rgb("rgb L", &[1.0, 1.0, 1.0]);
    params.add_float("float scale", f32::MAX);
    params.add_float("float illuminance", f32::MAX);
    params.add_point(
        "point portal",
        &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    );
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("portal", "infinite", params));

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error.to_string().contains("non-finite scale"));
}

#[test]
fn flatten_node_rejects_non_finite_final_uniform_infinite_scale() {
    let mut params = ParameterDictionary::default();
    params.add_rgb("rgb L", &[1.0, 1.0, 1.0]);
    params.add_float("float scale", f32::MAX);
    params.add_float("float illuminance", f32::MAX);
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("environment", "infinite", params));

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error.to_string().contains("non-finite scale"));
}

#[test]
fn flatten_node_rejects_non_finite_final_image_infinite_scale() {
    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("white.png");
    ImageBuffer::<Rgb<u8>, _>::from_pixel(3, 3, Rgb([255, 255, 255]))
        .save(&image_path)
        .unwrap();
    let mut params = ParameterDictionary::default();
    params.add_string("string filename", image_path.to_str().unwrap());
    params.add_string("string encoding", "linear");
    params.add_float("float scale", f32::MAX);
    params.add_float("float illuminance", f32::MAX);
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("environment", "infinite", params));

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error.to_string().contains("non-finite scale"));
}

#[test]
fn flatten_node_reports_degenerate_portal_geometry() {
    let mut params = ParameterDictionary::default();
    params.add_rgb("rgb L", &[1.0, 1.0, 1.0]);
    params.add_point(
        "point portal",
        &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    );
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("portal", "infinite", params));

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error.to_string().contains("degenerate"));
}

#[test]
fn flatten_node_builds_portal_jacobian_weighted_sat() {
    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("white.png");
    ImageBuffer::<Rgb<u8>, _>::from_pixel(3, 3, Rgb([255, 255, 255]))
        .save(&image_path)
        .unwrap();
    let mut params = ParameterDictionary::default();
    params.add_string("string filename", image_path.to_str().unwrap());
    params.add_point(
        "point portal",
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    );
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("portal", "infinite", params));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    let center = scene.portal_distribution[4];
    assert!((center.function - PI.powi(2)).abs() < 1e-4);
    let total: f32 = scene
        .portal_distribution
        .iter()
        .map(|texel| texel.function)
        .sum();
    assert!((scene.portal_distribution[8].summed_area - total).abs() < 1e-4);
}

#[test]
fn flatten_node_rejects_l_and_filename_without_portal() {
    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("white.png");
    ImageBuffer::<Rgb<u8>, _>::from_pixel(1, 1, Rgb([255, 255, 255]))
        .save(&image_path)
        .unwrap();
    let mut params = ParameterDictionary::default();
    params.add_rgb("rgb L", &[1.0, 1.0, 1.0]);
    params.add_string("string filename", image_path.to_str().unwrap());
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("environment", "infinite", params));

    assert!(flatten_node(Arc::new(RwLock::new(root))).is_err());
}

#[test]
fn flatten_node_keeps_portal_points_in_world_space() {
    let mut params = ParameterDictionary::default();
    params.add_rgb("rgb L", &[1.0, 1.0, 1.0]);
    params.add_point(
        "point portal",
        &[0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    );
    let light = light_node("portal", "infinite", params);
    light.write().unwrap().transform.matrix[3] = 10.0;
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.portal_infinite_lights[0].portal[0], [0.0, 0.0, 2.0]);
    assert_eq!(
        scene.light_sampling_models[0].world_to_light,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    );
}

#[test]
fn portal_distribution_preserves_negative_v4_weights() {
    let source =
        build_linear_rgb_mipmap([3, 3], &[[-1.0, -1.0, -1.0]; 9], ColorSpace::Srgb).unwrap();
    let prepared = prepare_portal_image(
        &source,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    )
    .unwrap();
    assert!((prepared.distribution[4].function + PI.powi(2)).abs() < 1e-4);
}

#[test]
fn flatten_node_prepares_equal_area_infinite_image_and_transform() {
    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("environment.png");
    ImageBuffer::<Rgb<u8>, _>::from_raw(2, 2, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        .unwrap()
        .save(&image_path)
        .unwrap();
    let mut params = ParameterDictionary::default();
    params.add_string("string filename", image_path.to_str().unwrap());
    params.add_string("string encoding", "linear");
    let light = light_node("environment", "infinite", params);
    light.write().unwrap().transform.matrix = [
        2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    let light = &scene.infinite_lights[0];
    let model = &scene.light_sampling_models[light.sampling_model as usize];
    assert_eq!(model.world_to_light[0][0], 0.5);
    assert_eq!(model.world_to_light[1][1], 0.25);
    assert_eq!(model.world_to_light[2][2], 0.2);
    let mipmap = &scene.texture_library.mipmaps[light.image_index as usize];
    assert_eq!(mipmap.levels[0].resolution, [2, 2]);
    assert_eq!(mipmap.levels[0].channels, 3);
    assert_eq!(light.attributes.len(), 3);
    assert_eq!(light.attributes[2].name, "image-illuminant");
    let actual =
        evaluate_dense_spectrum(&scene.spectrum_attributes, light.attributes[2].index, 450.0)
            .unwrap();
    assert!((actual - SRGB.illuminant.sample_at(450.0)).abs() < 1e-6);
}

#[test]
fn flatten_node_rejects_non_square_infinite_image() {
    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("environment.png");
    ImageBuffer::<Rgb<u8>, _>::from_raw(2, 1, vec![255, 0, 0, 0, 255, 0])
        .unwrap()
        .save(&image_path)
        .unwrap();
    let mut params = ParameterDictionary::default();
    params.add_string("string filename", image_path.to_str().unwrap());
    params.add_string("string encoding", "linear");
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("environment", "infinite", params));

    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error.to_string().contains("non-square resolution"));
}

#[test]
fn flatten_node_allows_multiple_infinite_lights() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(light_node("distant-a", "distant", Default::default()));
    root.add_child(light_node("distant-b", "distant", Default::default()));
    root.add_child(light_node("uniform", "infinite", Default::default()));
    root.add_child(light_node("duplicate", "infinite", Default::default()));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.infinite_lights.len(), 4);
    assert_eq!(scene.light_sampling_models.len(), 4);
    scene.validate_static_views().unwrap();
}

#[test]
fn flatten_node_uses_color_space_illuminant_for_default_light_spectra() {
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());

    let mut spot_params = ParameterDictionary::default();
    spot_params.set_color_space(&ACES2065_1);
    root.add_child(light_node("spot", "spot", spot_params));

    let mut distant_params = ParameterDictionary::default();
    distant_params.set_color_space(&ACES2065_1);
    root.add_child(light_node("distant", "distant", distant_params));

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    let expected = ACES2065_1.illuminant.sample_at(450.0);
    for light in [&scene.lights[0], &scene.infinite_lights[0]] {
        let actual =
            evaluate_dense_spectrum(&scene.spectrum_attributes, light.attributes[0].index, 450.0)
                .unwrap();
        assert!((actual - expected).abs() < 1e-6);
    }
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
    assert_eq!(scene.material_nodes[0].kind, "diffuse");
    assert_eq!(scene.material_nodes[0].attributes.len(), 1);
    assert_eq!(
        scene.material_nodes[0].attributes[0].kind,
        AttributeKind::Spectrum
    );
    assert_eq!(scene.material_nodes[0].attributes[0].name, "reflectance");
    let attribute = &scene.material_nodes[0].attributes[0];
    let base = attribute.index as usize * pbrt_r4::gpu::flat::DENSE_SAMPLE_COUNT;
    assert!(
        scene.spectrum_attributes[attribute.index as usize].samples[..3]
            .iter()
            .all(|v| v.is_finite())
    );
}

#[test]
fn flatten_node_extracts_diffuse_transmission_attributes() {
    let shape = triangle_node("triangle", "diffusetransmission", [0.0, 0.0, 0.0]);
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    let material = &scene.material_nodes[0];
    assert_eq!(material.kind, "diffusetransmission");
    assert_eq!(material.attributes.len(), 3);
    assert_eq!(material.attributes[0].name, "reflectance");
    assert_eq!(material.attributes[1].name, "transmittance");
    assert_eq!(material.attributes[2].name, "scale");
    assert_eq!(
        scene.scalar_attributes[material.attributes[2].index as usize],
        1.0
    );
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
    assert_eq!(scene.material_nodes[0].attributes.len(), 4);
    assert_eq!(
        scene.material_nodes[0].attributes[0].kind,
        AttributeKind::Spectrum
    );
    assert_eq!(scene.material_nodes[0].attributes[0].name, "eta");
    assert_eq!(scene.material_nodes[0].attributes[1].name, "uroughness");
    assert_eq!(scene.material_nodes[0].attributes[2].name, "vroughness");
    assert_eq!(scene.material_nodes[0].attributes[3].name, "remaproughness");
    let attribute = &scene.material_nodes[0].attributes[0];
    assert!(
        (evaluate_dense_spectrum(&scene.spectrum_attributes, attribute.index, 550.0).unwrap()
            - 1.33)
            .abs()
            < 1e-5
    );
}

#[test]
fn flatten_node_extracts_thin_dielectric_leaf() {
    let shape = triangle_node("triangle", "thindielectric", [0.0, 0.0, 0.0]);
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);
    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.material_nodes[0].attributes.len(), 1);
    assert_eq!(
        scene.material_nodes[0].attributes[0].kind,
        AttributeKind::Spectrum
    );
    assert_eq!(scene.material_nodes[0].attributes[0].name, "eta");
}

#[test]
fn flatten_node_expands_coateddiffuse_children() {
    let shape = triangle_node("coated", "coateddiffuse", [0.0; 3]);
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();

    let layout = scene.material_roots[scene.instances[0].material_root as usize];
    let material = &scene.material_nodes[layout.node_offset as usize];
    assert_eq!(material.source_kind, "coateddiffuse");
    assert_eq!(material.kind, "coateddiffuse");
    assert_eq!(material.attributes.len(), 10);
    assert_eq!(material.attributes[0].name, "thickness");
    assert_eq!(material.attributes[1].name, "reflectance");
    assert_eq!(material.attributes[2].name, "g");
    assert_eq!(material.attributes[3].name, "maxdepth");
    assert_eq!(material.attributes[4].name, "nsamples");
    assert_eq!(material.attributes[5].name, "albedo");
    assert_eq!(material.attributes[6].name, "eta");
    assert_eq!(material.attributes[7].name, "uroughness");
    assert_eq!(material.attributes[8].name, "vroughness");
    assert_eq!(material.attributes[9].name, "remaproughness");
    assert_ne!(material.child0, pbrt_r4::gpu::flat::INVALID_INDEX);
    assert_ne!(material.child1, pbrt_r4::gpu::flat::INVALID_INDEX);
    assert_eq!(
        pbrt_r4::gpu::flat::max_attributes_eval_work_items_per_surface(&scene).unwrap(),
        3
    );
}

#[test]
fn flatten_node_preserves_coatedconductor_layer_parameters() {
    let shape = triangle_node("coated", "coatedconductor", [0.0; 3]);
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
            .unwrap()
            .params
            .add_rgb("rgb reflectance", &[0.25, 0.5, 0.75]);
    }
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    let layout = scene.material_roots[scene.instances[0].material_root as usize];
    let material = &scene.material_nodes[layout.node_offset as usize];
    assert_eq!(material.kind, "coatedconductor");
    assert_eq!(material.attributes.len(), 15);
    assert_eq!(material.attributes[5].name, "interface.eta");
    assert_eq!(material.attributes[8].name, "conductor.eta");
    assert_eq!(material.attributes[9].name, "conductor.k");
    assert_eq!(material.attributes[13].name, "reflectance");
    assert_eq!(material.attributes[14].name, "use_reflectance");
    assert_eq!(
        scene.scalar_attributes[material.attributes[14].index as usize],
        1.0,
    );
    assert_eq!(
        scene.material_nodes[material.child1 as usize].kind,
        "conductor_reflectance"
    );
    assert_eq!(
        scene.material_nodes[material.child1 as usize]
            .attributes
            .len(),
        2
    );
}

#[test]
fn flatten_node_rejects_coated_layer_limits() {
    let shape = triangle_node("coated", "coateddiffuse", [0.0; 3]);
    {
        let mut node = shape.write().unwrap();
        if let Some(Component::Material(material)) = node
            .components
            .iter_mut()
            .find(|component| matches!(component, Component::Material(_)))
        {
            Arc::get_mut(&mut material.material)
                .unwrap()
                .params
                .add_int("maxdepth", 33);
        }
    }
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);
    let error = flatten_node(Arc::new(RwLock::new(root))).unwrap_err();
    assert!(error.to_string().contains("maxdepth 33"));
}

#[test]
fn material_table_uses_generic_attribute_ranges() {
    use pbrt_r4::gpu::webgpu::material::MaterialTable;
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    for kind in ["diffuse", "dielectric", "diffuse", "thindielectric"] {
        root.add_child(triangle_node(kind, kind, [0.0; 3]));
    }
    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    let table = MaterialTable::from_flat(&scene).unwrap();
    assert_eq!(table.nodes.len(), scene.material_nodes.len());
    assert_eq!(
        table.attributes.len(),
        scene
            .material_nodes
            .iter()
            .map(|m| m.attributes.len())
            .sum::<usize>()
    );
    for record in &table.nodes {
        assert!(
            (record.attribute_offset as usize) + (record.attribute_count as usize)
                <= table.attributes.len()
        );
    }
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
fn flatten_node_extracts_conductor_attributes() {
    let shape = triangle_node("triangle", "conductor_eta_k", [0.0, 0.0, 0.0]);
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.material_nodes[0].kind, "conductor_eta_k");
    assert_eq!(scene.material_nodes[0].attributes.len(), 3);
    for attribute in &scene.material_nodes[0].attributes[..2] {
        let base = attribute.index as usize * pbrt_r4::gpu::flat::DENSE_SAMPLE_COUNT;
        assert!(
            scene.spectrum_attributes[attribute.index as usize].samples[..3]
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
        );
    }
    assert_eq!(scene.scalar_attributes[0], 0.0);
}

#[test]
fn flatten_node_keeps_conductor_reflectance_layout_separate() {
    let shape = triangle_node("triangle", "conductor_reflectance", [0.0, 0.0, 0.0]);
    {
        let mut node = shape.write().unwrap();
        let Component::Material(material) = &mut node.components[1] else {
            panic!("expected material component");
        };
        Arc::get_mut(&mut material.material)
            .expect("test material should be uniquely owned")
            .params
            .add_rgb("rgb reflectance", &[0.5, 0.5, 0.5]);
    }
    let mut root = Node::new("root");
    add_camera_and_film(&mut root, Default::default());
    root.add_child(shape);

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.material_nodes[0].kind, "conductor_reflectance");
    assert_eq!(scene.material_nodes[0].attributes.len(), 2);
    assert_eq!(scene.material_nodes[0].attributes[0].name, "reflectance");
    assert_eq!(scene.material_nodes[0].attributes[1].name, "roughness");
}

#[test]
fn node_ir_preparation_completes_missing_mesh_uvs_before_flattening() {
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
    tessellate_shapes(&mut root).unwrap();
    prepare_triangle_meshes(&mut root).unwrap();

    let scene = flatten_node(Arc::new(RwLock::new(root))).unwrap();
    assert_eq!(scene.vertices.len(), 3);
    assert_eq!(scene.vertices[0].uv, [0.0, 0.0]);
    assert_eq!(scene.vertices[1].uv, [1.0, 0.0]);
    assert_eq!(scene.vertices[2].uv, [0.0, 1.0]);
    // Missing normal/tangent are left zero: shade_surface.wgsl recomputes a
    // fresh per-triangle geometric normal / dpdu whenever the interpolated
    // vertex value is zero, so there is nothing to precompute here.
    assert_eq!(scene.vertices[0].normal, [0.0, 0.0, 0.0]);
    assert_eq!(scene.vertices[0].tangent, [0.0, 0.0, 0.0]);
}

#[test]
fn missing_normals_are_left_for_shade_surface_to_recompute() {
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
    assert_eq!(completed.positions.len(), 4);
    assert_eq!(completed.indices, vec![0, 1, 2, 0, 2, 3]);
    assert!(completed.normals.is_none());
    assert!(completed.tangents.is_none());
}
