#![cfg(feature = "webgpu")]

use std::collections::HashMap;

use pbrt_r4::gpu::flat::{self, AttributeKind, AttributeRef as FlatAttributeRef};
use pbrt_r4::gpu::webgpu::abi::AttributeRef;
use pbrt_r4::gpu::webgpu::abi::{
    LIGHT_KIND_AREA, LIGHT_KIND_DISTANT, LIGHT_KIND_IMAGE_INFINITE, LIGHT_KIND_POINT,
    LIGHT_KIND_PORTAL_IMAGE_INFINITE, LIGHT_KIND_SPOT, LIGHT_KIND_UNIFORM_INFINITE,
};
use pbrt_r4::gpu::webgpu::scene::light::convert_lights;

fn flat_attribute(kind: AttributeKind, index: u32) -> FlatAttributeRef {
    FlatAttributeRef {
        kind,
        index,
        name: String::new(),
    }
}

fn sampling_model(
    kind: flat::LightKind,
    geometry_kind: flat::LightGeometryKind,
    flags: u32,
) -> flat::LightSamplingModel {
    flat::LightSamplingModel {
        kind,
        geometry_kind,
        geometry_index: 4,
        direction_index: 5,
        distribution_offset: 6,
        distribution_count: 7,
        total_area: 8.0,
        flags,
        world_to_light: [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ],
    }
}

#[test]
fn converts_light_records_and_sampling_models_in_flat_order() {
    let lights = vec![flat::Light {
        kind: flat::LightKind::Point,
        attributes: vec![flat_attribute(AttributeKind::Scalar, 11)],
        sampling_model: 0,
        image_index: flat::INVALID_INDEX,
    }];
    let infinite_lights = vec![flat::Light {
        kind: flat::LightKind::ImageInfinite,
        attributes: vec![flat_attribute(AttributeKind::Spectrum, 12)],
        sampling_model: 1,
        image_index: 9,
    }];
    let models = [
        sampling_model(flat::LightKind::Point, flat::LightGeometryKind::Position, 3),
        sampling_model(
            flat::LightKind::ImageInfinite,
            flat::LightGeometryKind::ImageInfinite,
            5,
        ),
    ];
    let image_payloads = HashMap::from([(1, 42)]);
    let existing_attribute = AttributeRef { kind: 3, index: 8 };

    let data = convert_lights(
        &models,
        &lights,
        &infinite_lights,
        vec![existing_attribute],
        &image_payloads,
    )
    .unwrap();

    assert_eq!(data.attributes.len(), 3);
    assert_eq!(data.attributes[0].kind, 3);
    assert_eq!(data.attributes[0].index, 8);
    assert_eq!(data.attributes[1].kind, 0);
    assert_eq!(data.attributes[1].index, 11);
    assert_eq!(data.attributes[2].kind, 1);
    assert_eq!(data.attributes[2].index, 12);

    assert_eq!(data.records.len(), 2);
    assert_eq!(data.records[0].kind, LIGHT_KIND_POINT);
    assert_eq!(data.records[0].attribute_offset, 1);
    assert_eq!(data.records[0].attribute_count, 1);
    assert_eq!(data.records[0].sampling_model, 0);
    assert_eq!(data.records[1].kind, LIGHT_KIND_IMAGE_INFINITE);
    assert_eq!(data.records[1].attribute_offset, 2);
    assert_eq!(data.records[1].attribute_count, 1);
    assert_eq!(data.records[1].sampling_model, 1);

    assert_eq!(data.sampling_models[0].kind, LIGHT_KIND_POINT);
    assert_eq!(data.sampling_models[0].geometry_kind, 0);
    assert_eq!(data.sampling_models[0].flags, 3);
    assert_eq!(data.sampling_models[1].kind, LIGHT_KIND_IMAGE_INFINITE);
    assert_eq!(data.sampling_models[1].geometry_kind, 4);
    assert_eq!(data.sampling_models[1].flags, 42);
    assert_eq!(
        data.sampling_models[1].world_to_light,
        models[1].world_to_light
    );
}

#[test]
fn maps_every_flat_light_and_geometry_kind() {
    let cases = [
        (
            flat::LightKind::Point,
            LIGHT_KIND_POINT,
            flat::LightGeometryKind::Position,
            0,
        ),
        (
            flat::LightKind::Spot,
            LIGHT_KIND_SPOT,
            flat::LightGeometryKind::Instance,
            1,
        ),
        (
            flat::LightKind::Area,
            LIGHT_KIND_AREA,
            flat::LightGeometryKind::Direction,
            2,
        ),
        (
            flat::LightKind::Distant,
            LIGHT_KIND_DISTANT,
            flat::LightGeometryKind::Portal,
            3,
        ),
        (
            flat::LightKind::UniformInfinite,
            LIGHT_KIND_UNIFORM_INFINITE,
            flat::LightGeometryKind::ImageInfinite,
            4,
        ),
        (
            flat::LightKind::ImageInfinite,
            LIGHT_KIND_IMAGE_INFINITE,
            flat::LightGeometryKind::Position,
            0,
        ),
        (
            flat::LightKind::PortalImageInfinite,
            LIGHT_KIND_PORTAL_IMAGE_INFINITE,
            flat::LightGeometryKind::Instance,
            1,
        ),
    ];
    let models = cases
        .iter()
        .map(|(kind, _, geometry_kind, _)| sampling_model(*kind, *geometry_kind, 0))
        .collect::<Vec<_>>();

    let data = convert_lights(&models, &[], &[], Vec::new(), &HashMap::new()).unwrap();

    for (model, (_, expected_kind, _, expected_geometry_kind)) in
        data.sampling_models.iter().zip(cases)
    {
        assert_eq!(model.kind, expected_kind);
        assert_eq!(model.geometry_kind, expected_geometry_kind);
    }
}
