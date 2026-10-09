#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::{identity_transform, Medium};
use pbrt_r4::gpu::webgpu::scene::medium::convert_media;

fn homogeneous_medium() -> Medium {
    Medium {
        name: "fog".to_owned(),
        kind: "homogeneous".to_owned(),
        sigma_a: 0,
        sigma_s: 1,
        le: 2,
        g: 0.25,
        transform: identity_transform(),
    }
}

#[test]
fn converts_homogeneous_medium_to_record() {
    let record = convert_media(&[homogeneous_medium()], 3).unwrap()[0];

    assert_eq!(record.kind, 0);
    assert_eq!([record.sigma_a, record.sigma_s, record.le], [0, 1, 2]);
    assert_eq!(record.g, 0.25);
    assert_eq!(record.padding, [0; 3]);
    assert_eq!(
        record.medium_to_world,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
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
