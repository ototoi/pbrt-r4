#![cfg(feature = "webgpu")]

use std::sync::{Arc, RwLock};

use pbrt_r4::gpu::flat::portal::{
    PortalDistributionTexel as FlatPortalDistributionTexel,
    PortalImageInfiniteLight as FlatPortalImageInfiniteLight,
};
use pbrt_r4::gpu::flat::{
    self, ImageInfiniteDistributionTexel as FlatImageInfiniteDistributionTexel,
    ImageInfiniteSamplingRecord as FlatImageInfiniteSamplingRecord,
    TriangleDistributionEntry as FlatTriangleDistributionEntry,
};
use pbrt_r4::gpu::node::{
    Camera, CameraComponent, Component, Film, FilmComponent, Node, Output, OutputComponent,
};
use pbrt_r4::gpu::webgpu::scene::light::LightSamplingData;
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

#[test]
fn converts_light_sampling_payload_records_without_reordering() {
    let mut scene = empty_flat_scene();
    scene.light_positions = vec![[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]];
    scene.triangle_distributions = vec![
        FlatTriangleDistributionEntry {
            primitive: 4,
            cdf: 0.25,
            area: 2.0,
        },
        FlatTriangleDistributionEntry {
            primitive: 7,
            cdf: 1.0,
            area: 6.0,
        },
    ];
    scene.portal_infinite_lights = vec![FlatPortalImageInfiniteLight {
        portal: [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        world_to_portal: [[25.0; 4], [26.0; 4], [27.0; 4]],
        distribution_offset: 13,
        resolution: [14, 15],
    }];
    scene.portal_distribution = vec![FlatPortalDistributionTexel {
        function: 16.0,
        summed_area: 17.0,
    }];
    scene.image_infinite_lights = vec![FlatImageInfiniteSamplingRecord {
        distribution_offset: 18,
        row_cdf_offset: 19,
        resolution: [20, 21],
        light_to_render: [[22.0; 4]; 3],
    }];
    scene.image_infinite_distribution = vec![FlatImageInfiniteDistributionTexel {
        weight: 23.0,
        conditional_cdf: 24.0,
    }];

    let data = LightSamplingData::from_flat(&scene);

    assert_eq!(
        data.positions,
        [[1.0, 2.0, 3.0, 1.0], [-1.0, -2.0, -3.0, 1.0]]
    );
    assert_eq!(data.triangle_distributions.len(), 2);
    assert_eq!(data.triangle_distributions[0].primitive, 4);
    assert_eq!(data.triangle_distributions[0].cdf, 0.25);
    assert_eq!(data.triangle_distributions[0].area, 2.0);
    assert_eq!(data.triangle_distributions[0].reserved, 0);
    assert_eq!(data.triangle_distributions[1].primitive, 7);
    assert_eq!(data.triangle_distributions[1].cdf, 1.0);
    assert_eq!(data.triangle_distributions[1].area, 6.0);
    assert_eq!(data.portal_records.len(), 1);
    assert_eq!(
        data.portal_records[0].portal,
        [
            [1.0, 2.0, 3.0, 1.0],
            [4.0, 5.0, 6.0, 1.0],
            [7.0, 8.0, 9.0, 1.0],
            [10.0, 11.0, 12.0, 1.0],
        ]
    );
    assert_eq!(
        data.portal_records[0].world_to_portal,
        [[25.0; 4], [26.0; 4], [27.0; 4]]
    );
    assert_eq!(data.portal_records[0].distribution_offset, 13);
    assert_eq!(
        [data.portal_records[0].width, data.portal_records[0].height],
        [14, 15]
    );
    assert_eq!(data.portal_records[0].reserved, 0);
    assert_eq!(data.portal_distribution.len(), 1);
    assert_eq!(data.portal_distribution[0].function, 16.0);
    assert_eq!(data.portal_distribution[0].summed_area, 17.0);
    assert_eq!(data.image_infinite_records.len(), 1);
    assert_eq!(data.image_infinite_records[0].distribution_offset, 18);
    assert_eq!(data.image_infinite_records[0].row_cdf_offset, 19);
    assert_eq!(
        [
            data.image_infinite_records[0].width,
            data.image_infinite_records[0].height
        ],
        [20, 21]
    );
    assert_eq!(
        data.image_infinite_records[0].light_to_render,
        [[22.0; 4]; 3]
    );
    assert_eq!(data.image_infinite_distribution.len(), 1);
    assert_eq!(data.image_infinite_distribution[0].weight, 23.0);
    assert_eq!(data.image_infinite_distribution[0].conditional_cdf, 24.0);
}
