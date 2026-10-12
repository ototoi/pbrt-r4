#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::sparse_grid::{SparseGridEntry, SparseGridRoot, SparseScalarGrid};
use pbrt_r4::gpu::flat::{
    CloudMedium, ColorSpace, Medium, MediumData, NanoVDBMedium, RgbGridMedium, UniformGridMedium,
};
use pbrt_r4::gpu::webgpu::scene::medium::{convert_media, validate_storage_array_size};

fn homogeneous_medium() -> Medium {
    Medium {
        name: "fog".to_owned(),
        kind: "homogeneous".to_owned(),
        sigma_a: 0,
        sigma_s: 1,
        le: 2,
        g: 0.25,
        data: MediumData::Homogeneous,
        transform: [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    }
}

fn nanovdb_medium() -> Medium {
    let mut medium = homogeneous_medium();
    medium.name = "sparse fog".to_owned();
    medium.kind = "nanovdb".to_owned();
    medium.data = MediumData::NanoVDB(NanoVDBMedium {
        density: SparseScalarGrid {
            roots: vec![SparseGridRoot {
                block: [0; 3],
                entry: SparseGridEntry {
                    child_index: None,
                    value: 0.5,
                },
            }],
            ..Default::default()
        },
        index_bbox_min: [0; 3],
        index_bbox_max: [7; 3],
        inverse_map_f: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_f: [0.0; 3],
        bounds_min: [0.0; 3],
        bounds_max: [1.0; 3],
        majorant_resolution: [64; 3],
        majorant: vec![0.5; 64usize.pow(3)],
    });
    medium
}

#[test]
fn converts_homogeneous_medium_to_record() {
    let converted = convert_media(&[homogeneous_medium()], 3).unwrap();
    let record = converted.records[0];

    assert_eq!(record.kind, 0);
    assert_eq!([record.sigma_a, record.sigma_s, record.le], [0, 1, 2]);
    assert_eq!(record.g, 0.25);
    assert_eq!(record.volume_index, u32::MAX);
    assert_eq!(record.padding, [0; 2]);
    assert_eq!(
        record.medium_to_world,
        [
            [0.0, 4.0, 8.0, 12.0],
            [1.0, 5.0, 9.0, 13.0],
            [2.0, 6.0, 10.0, 14.0],
            [3.0, 7.0, 11.0, 15.0],
        ]
    );
}

#[test]
fn rejects_invalid_spectrum_references() {
    let mut medium = homogeneous_medium();
    medium.le = 3;

    assert!(convert_media(&[medium], 3).is_err());
}

#[test]
fn validates_storage_buffer_device_limits_and_boundaries() {
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = 8;
    limits.max_buffer_size = 8;

    assert!(validate_storage_array_size::<u32>(&limits, 2, "probe").is_ok());
    assert!(validate_storage_array_size::<u32>(&limits, 1, "probe").is_ok());

    let binding_error = validate_storage_array_size::<u32>(&limits, 3, "probe").unwrap_err();
    assert!(binding_error
        .to_string()
        .contains("storage binding limit is 8 bytes"));

    limits.max_storage_buffer_binding_size = 16;
    let buffer_error = validate_storage_array_size::<u32>(&limits, 3, "probe").unwrap_err();
    assert!(buffer_error
        .to_string()
        .contains("device buffer limit is 8 bytes"));
}

#[test]
fn rejects_unsupported_medium_kinds() {
    let mut medium = homogeneous_medium();
    medium.kind = "heterogeneous".to_owned();

    assert!(convert_media(&[medium], 3).is_err());
}

#[test]
fn packs_uniformgrid_data_and_assigns_checked_ranges() {
    let mut medium = homogeneous_medium();
    medium.kind = "uniformgrid".to_owned();
    medium.transform = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    medium.data = MediumData::UniformGrid(UniformGridMedium {
        bounds_min: [0.0; 3],
        bounds_max: [1.0; 3],
        resolution: [2, 1, 1],
        density: vec![0.25, 1.0],
        majorant_resolution: [1; 3],
        majorant: vec![1.0],
    });
    let converted = convert_media(&[homogeneous_medium(), medium], 3).unwrap();
    assert_eq!(converted.records[0].kind, 0);
    assert_eq!(converted.records[0].volume_index, u32::MAX);
    assert_eq!(converted.records[1].kind, 1);
    assert_eq!(converted.records[1].volume_index, 0);
    assert_eq!(converted.volumes.len(), 1);
    assert_eq!(converted.volumes[0].density_offset_count, [0, 2, 0, 0]);
    assert_eq!(converted.volumes[0].majorant_offset_count, [2, 1, 0, 0]);
    assert_eq!(converted.volume_data, [0.25, 1.0, 1.0]);
}

#[test]
fn packs_cloud_parameters_and_unit_majorant() {
    let mut medium = homogeneous_medium();
    medium.kind = "cloud".to_owned();
    medium.transform = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    medium.data = MediumData::Cloud(CloudMedium {
        bounds_min: [-1.0, 0.0, -2.0],
        bounds_max: [1.0, 3.0, 2.0],
        density: 0.75,
        wispiness: 0.5,
        frequency: 4.0,
    });

    let converted = convert_media(&[medium], 3).unwrap();

    assert_eq!(converted.records[0].kind, 2);
    assert_eq!(converted.records[0].volume_index, 0);
    assert_eq!(converted.volumes.len(), 1);
    assert_eq!(converted.volumes[0].bounds_min, [-1.0, 0.0, -2.0, 0.0]);
    assert_eq!(converted.volumes[0].bounds_max, [1.0, 3.0, 2.0, 0.0]);
    assert_eq!(converted.volumes[0].resolution, [1, 1, 1, 0]);
    assert_eq!(converted.volumes[0].majorant_resolution, [1, 1, 1, 0]);
    assert_eq!(converted.volumes[0].majorant_offset_count, [0, 1, 0, 0]);
    assert_eq!(converted.volumes[0].cloud_parameters, [0.75, 0.5, 4.0, 0.0]);
    assert_eq!(converted.volume_data, [1.0]);
}

#[test]
fn packs_rgbgrid_coefficients_scale_color_space_and_majorants() {
    let mut medium = homogeneous_medium();
    medium.kind = "rgbgrid".to_owned();
    medium.transform = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    medium.data = MediumData::RgbGrid(RgbGridMedium {
        bounds_min: [0.0; 3],
        bounds_max: [1.0; 3],
        resolution: [2, 1, 1],
        sigma_a: Some(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        sigma_s: Some(vec![[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]),
        scale: 2.0,
        color_space: ColorSpace::Aces2065,
        majorant_resolution: [16; 3],
        majorant: vec![3.0; 16 * 16 * 16],
    });

    let converted = convert_media(&[medium], 0).unwrap();

    assert_eq!(converted.records[0].kind, 3);
    assert_eq!(converted.records[0].volume_index, 0);
    let record = converted.volumes[0];
    assert_eq!(record.resolution, [2, 1, 1, 1]);
    assert_eq!(record.density_offset_count, [0, 2, 0, 0]);
    assert_eq!(record.coefficient_offset_count, [6, 2, 0, 0]);
    assert_eq!(record.majorant_offset_count, [12, 4096, 0, 0]);
    assert_eq!(record.cloud_parameters, [2.0, 0.0, 0.0, 0.0]);
    assert_eq!(converted.volume_data.len(), 12 + 4096);
    assert_eq!(&converted.volume_data[..6], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    assert_eq!(
        &converted.volume_data[6..12],
        &[0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    );
    assert!(converted.volume_data[12..]
        .iter()
        .all(|value| *value == 3.0));
}

#[test]
fn packs_missing_rgbgrid_coefficient_as_invalid_range() {
    let mut medium = homogeneous_medium();
    medium.kind = "rgbgrid".to_owned();
    medium.transform = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    medium.data = MediumData::RgbGrid(RgbGridMedium {
        bounds_min: [0.0; 3],
        bounds_max: [1.0; 3],
        resolution: [1; 3],
        sigma_a: None,
        sigma_s: Some(vec![[0.5; 3]]),
        scale: 1.0,
        color_space: ColorSpace::Srgb,
        majorant_resolution: [16; 3],
        majorant: vec![2.0; 16 * 16 * 16],
    });
    let converted = convert_media(&[medium], 0).unwrap();
    assert_eq!(
        converted.volumes[0].density_offset_count,
        [u32::MAX, 0, 0, 0]
    );
    assert_eq!(converted.volumes[0].coefficient_offset_count, [0, 1, 0, 0]);
}

#[test]
fn accepts_rotated_reflected_nonuniform_nanovdb_medium_transform() {
    let mut medium = nanovdb_medium();
    medium.transform = [
        0.0, -2.0, 0.0, 3.0, 3.0, 0.0, 0.0, -4.0, 0.0, 0.0, -0.5, 5.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let converted = convert_media(&[medium], 3).unwrap();
    let inverse = converted.volumes[0].medium_from_world;
    let expected = [
        [0.0, 1.0 / 3.0, 0.0, 4.0 / 3.0],
        [-0.5, 0.0, 0.0, 1.5],
        [0.0, 0.0, -2.0, 10.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for column in 0..4 {
        for row in 0..4 {
            assert!((inverse[column][row] - expected[row][column]).abs() < 1.0e-6);
        }
    }
}

#[test]
fn rejects_malformed_rgbgrid_upload_payloads() {
    let rgb_grid_medium = || {
        let mut medium = homogeneous_medium();
        medium.kind = "rgbgrid".to_owned();
        medium.transform = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        medium.data = MediumData::RgbGrid(RgbGridMedium {
            bounds_min: [0.0; 3],
            bounds_max: [1.0; 3],
            resolution: [1; 3],
            sigma_a: Some(vec![[0.5; 3]]),
            sigma_s: None,
            scale: 1.0,
            color_space: ColorSpace::Srgb,
            majorant_resolution: [16; 3],
            majorant: vec![1.0; 16 * 16 * 16],
        });
        medium
    };

    let mut invalid_majorant_count = rgb_grid_medium();
    let MediumData::RgbGrid(grid) = &mut invalid_majorant_count.data else {
        unreachable!();
    };
    grid.majorant.pop();
    assert!(convert_media(&[invalid_majorant_count], 0).is_err());

    let mut invalid_majorant_value = rgb_grid_medium();
    let MediumData::RgbGrid(grid) = &mut invalid_majorant_value.data else {
        unreachable!();
    };
    grid.majorant[0] = f32::NAN;
    assert!(convert_media(&[invalid_majorant_value], 0).is_err());

    let mut missing_coefficients = rgb_grid_medium();
    let MediumData::RgbGrid(grid) = &mut missing_coefficients.data else {
        unreachable!();
    };
    grid.sigma_a = None;
    assert!(convert_media(&[missing_coefficients], 0).is_err());
}
