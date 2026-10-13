#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::{AttributeKind, AttributeRef, BSSRDFCoefficientKind, BSSRDF};
use pbrt_r4::gpu::webgpu::bssrdf::convert_bssrdf_materials;

fn bssrdf(coefficient_kind: BSSRDFCoefficientKind, table_index: u32) -> BSSRDF {
    BSSRDF {
        scale: 0.75,
        g: 0.0,
        eta: 1.33,
        table_index,
        coefficient_kind,
        coefficients: [
            AttributeRef {
                kind: AttributeKind::Spectrum,
                index: 4,
                name: "sigma_a".to_owned(),
            },
            AttributeRef {
                kind: AttributeKind::Texture,
                index: 7,
                name: "sigma_s".to_owned(),
            },
        ],
    }
}

#[test]
fn converts_bssrdf_material_coefficient_kinds_to_record_tags() {
    let records = convert_bssrdf_materials(
        &[
            bssrdf(BSSRDFCoefficientKind::Sigma, 0),
            bssrdf(BSSRDFCoefficientKind::ReflectanceMfp, 1),
        ],
        2,
    )
    .unwrap();

    assert_eq!(records.len(), 2);
    assert_eq!(records[0].scale, 0.75);
    assert_eq!(records[0].eta, 1.33);
    assert_eq!(records[0].table_index, 0);
    assert_eq!(records[0].coefficient_kind, 0);
    assert_eq!(records[1].table_index, 1);
    assert_eq!(records[1].coefficient_kind, 1);
}

#[test]
fn rejects_bssrdf_materials_with_invalid_table_references() {
    assert!(convert_bssrdf_materials(&[bssrdf(BSSRDFCoefficientKind::Sigma, 1)], 1).is_err());
}
