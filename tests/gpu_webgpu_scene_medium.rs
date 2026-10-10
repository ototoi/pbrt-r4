#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::{Medium, MediumData, UniformGridMedium};
use pbrt_r4::gpu::webgpu::scene::medium::convert_media;

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

#[test]
fn converts_homogeneous_medium_to_record() {
    let converted = convert_media(&[homogeneous_medium()], 3).unwrap();
    let record = converted.records[0];

    assert_eq!(record.kind, 0);
    assert_eq!([record.sigma_a, record.sigma_s, record.le], [0, 1, 2]);
    assert_eq!(record.g, 0.25);
    assert_eq!(record.grid_index, u32::MAX);
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
    assert_eq!(converted.records[0].grid_index, u32::MAX);
    assert_eq!(converted.records[1].kind, 1);
    assert_eq!(converted.records[1].grid_index, 0);
    assert_eq!(converted.uniform_grids.len(), 1);
    assert_eq!(
        converted.uniform_grids[0].density_offset_count,
        [0, 2, 0, 0]
    );
    assert_eq!(
        converted.uniform_grids[0].majorant_offset_count,
        [2, 1, 0, 0]
    );
    assert_eq!(converted.volume_data, [0.25, 1.0, 1.0]);
}
