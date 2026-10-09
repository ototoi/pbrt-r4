#![cfg(feature = "webgpu")]

use std::sync::{Arc, RwLock};

use pbrt_r4::gpu::flat::{self, INVALID_INDEX};
use pbrt_r4::gpu::node::{
    Camera, CameraComponent, Component, Film, FilmComponent, Node, Output, OutputComponent,
};
use pbrt_r4::gpu::webgpu::scene::instance::convert_instances;
use pbrt_r4::paramdict::ParameterDictionary;

fn empty_flat_scene() -> flat::Scene {
    let mut root = Node::new("root");
    root.add_component(Component::Output(OutputComponent {
        output: Output {
            filename: "test.exr".to_owned(),
        },
    }));
    let mut camera = Node::new("camera");
    camera.add_component(Component::Camera(CameraComponent {
        camera: Camera {
            kind: "perspective".to_owned(),
            params: Default::default(),
            medium: None,
        },
    }));
    let mut film_params = ParameterDictionary::default();
    film_params.add_int("integer xresolution", 1);
    film_params.add_int("integer yresolution", 1);
    camera.add_component(Component::Film(FilmComponent {
        film: Film {
            name: "rgb".to_owned(),
            params: film_params,
        },
    }));
    root.add_child(Arc::new(RwLock::new(camera)));
    flat::flatten_node(Arc::new(RwLock::new(root))).unwrap()
}

fn instance() -> flat::Instance {
    flat::Instance {
        geometry: 0,
        transform: [
            2.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0, -8.0, 7.0, 0.0, 0.0, 0.0, 1.0,
        ],
        material_root: INVALID_INDEX,
        area_light: INVALID_INDEX,
        reverse_orientation: false,
        shape_transform_swaps_handedness: false,
        inside_medium: INVALID_INDEX,
        outside_medium: INVALID_INDEX,
    }
}

#[test]
fn preserves_instance_order_matrices_indices_and_orientation_bits() {
    let mut scene = empty_flat_scene();
    scene.geometries = vec![flat::Geometry::default(); 2];
    scene.material_roots = vec![
        flat::MaterialRoot {
            node_offset: 0,
            node_count: 1
        };
        2
    ];
    scene.media = (0..2)
        .map(|_| flat::Medium {
            name: String::new(),
            kind: "homogeneous".to_owned(),
            sigma_a: 0,
            sigma_s: 0,
            le: 0,
            g: 0.0,
            transform: instance().transform,
        })
        .collect();
    for reversed in [false, true] {
        for handedness in [false, true] {
            for shape_handedness in [false, true] {
                let mut value = instance();
                value.geometry = 1;
                value.material_root = 1;
                value.inside_medium = 0;
                value.outside_medium = 1;
                value.reverse_orientation = reversed;
                value.shape_transform_swaps_handedness = shape_handedness;
                if !handedness {
                    value.transform[10] = 8.0;
                }
                scene.instances.push(value);
            }
        }
    }
    let records = convert_instances(&scene, 2).unwrap();
    assert_eq!(records.len(), 8);
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record.orientation_flags, [0, 4, 2, 6, 1, 5, 3, 7][index]);
        let z = if index & 2 == 0 { 8.0 } else { -8.0 };
        assert_eq!(record.geometry, 1);
        assert_eq!(record.material_root, 1);
        assert_eq!(record.area_light, INVALID_INDEX);
        assert_eq!(record.medium_inside, 0);
        assert_eq!(record.medium_outside, 1);
        assert_eq!(record.padding, [0; 2]);
        assert_eq!(
            record.world_from_object,
            [
                [2.0, 0.0, 0.0, 0.0],
                [1.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, z, 0.0],
                [3.0, 5.0, 7.0, 1.0],
            ]
        );
        assert_eq!(
            record.normal_from_object,
            [
                [0.5, -0.125, 0.0, 0.0],
                [0.0, 0.25, 0.0, 0.0],
                [0.0, 0.0, 1.0 / z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
    }
    scene.instances = vec![instance()];
    let record = &convert_instances(&scene, 2).unwrap()[0];
    assert_eq!(record.material_root, INVALID_INDEX);
    assert_eq!(record.medium_inside, INVALID_INDEX);
    assert_eq!(record.medium_outside, INVALID_INDEX);
}

#[test]
fn validates_references_before_transform_in_existing_order() {
    let mut scene = empty_flat_scene();
    scene.geometries = vec![flat::Geometry {
        index_count: 3,
        ..Default::default()
    }];
    let mut value = instance();
    value.geometry = 1;
    value.inside_medium = 0;
    value.outside_medium = 0;
    value.material_root = 0;
    value.area_light = 0;
    value.transform[0] = 0.0;
    scene.instances = vec![value];
    let cases = [
        "references an invalid geometry",
        "references an invalid inside medium",
        "references an invalid outside medium",
        "references an invalid material",
        "has an invalid area-light handle",
        "transform is not invertible",
    ];
    for (index, expected) in cases.iter().enumerate() {
        let error = convert_instances(&scene, 1).err().unwrap().to_string();
        assert!(error.contains(expected), "{error}");
        match index {
            0 => scene.instances[0].geometry = 0,
            1 => scene.instances[0].inside_medium = INVALID_INDEX,
            2 => scene.instances[0].outside_medium = INVALID_INDEX,
            3 => scene.instances[0].material_root = INVALID_INDEX,
            4 => scene.instances[0].area_light = INVALID_INDEX,
            _ => (),
        }
    }
}

#[test]
fn accepts_empty_instances() {
    assert!(convert_instances(&empty_flat_scene(), 0)
        .unwrap()
        .is_empty());
}

#[test]
fn preserves_and_validates_area_light_links_at_the_instance_index() {
    let mut scene = empty_flat_scene();
    scene.geometries = vec![flat::Geometry {
        index_count: 3,
        ..Default::default()
    }];
    scene.instances = vec![instance(), instance()];
    scene.instances[1].area_light = 1;
    scene.lights.lights = vec![
        flat::Light {
            kind: flat::LightKind::Area,
            attributes: Vec::new(),
            sampling_model: 0,
            image_index: INVALID_INDEX,
        };
        2
    ];
    scene.lights.sampling_models = vec![flat::LightSamplingModel {
        kind: flat::LightKind::Area,
        geometry_kind: flat::LightGeometryKind::Instance,
        geometry_index: 1,
        direction_index: INVALID_INDEX,
        distribution_offset: 0,
        distribution_count: 1,
        total_area: 2.0,
        flags: 0,
        world_to_light: [[0.0; 4]; 3],
    }];
    scene.lights.triangle_distributions = vec![flat::TriangleDistributionEntry {
        primitive: 0,
        cdf: 1.0,
        area: 2.0,
    }];
    let records = convert_instances(&scene, 1).unwrap();
    assert_eq!(records[0].area_light, INVALID_INDEX);
    assert_eq!(records[1].area_light, 1);

    scene.lights.sampling_models[0].geometry_index = 0;
    let error = convert_instances(&scene, 1).err().unwrap().to_string();
    assert!(
        error.contains("Flat instance 1 area-light range does not match its triangles"),
        "{error}"
    );
}
