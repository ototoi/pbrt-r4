#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::{
    MeasuredAtlasPage, MeasuredBsdfRecord as FlatMeasuredBsdfRecord, MeasuredBsdfResources,
    MeasuredTableRecord as FlatMeasuredTableRecord,
};
use pbrt_r4::gpu::webgpu::measured::MeasuredBsdfRecords;

#[test]
fn converts_measured_bsdf_records_and_tables_to_webgpu_abi() {
    let resources = MeasuredBsdfResources {
        bsdfs: vec![
            FlatMeasuredBsdfRecord {
                ndf: 1,
                sigma: 2,
                vndf: 3,
                luminance: 4,
                spectra: 5,
                isotropic: true,
            },
            FlatMeasuredBsdfRecord {
                ndf: 6,
                sigma: 7,
                vndf: 8,
                luminance: 9,
                spectra: 10,
                isotropic: false,
            },
        ],
        tables: vec![FlatMeasuredTableRecord {
            size: [11, 12],
            parameter_count: 2,
            parameter_sizes: [13, 14, 15],
            parameter_strides: [16, 17, 18],
            parameter_value_offsets: [19, 20, 21],
            data_offset: 22,
            marginal_cdf_offset: 23,
            conditional_cdf_offset: 24,
        }],
        atlas_pages: vec![MeasuredAtlasPage {
            resolution: [1, 1],
            texels: vec![0.0; 4],
        }],
    };

    let records = MeasuredBsdfRecords::from_flat(&resources);

    assert_eq!(records.bsdfs.len(), 2);
    assert_eq!(records.bsdfs[0].ndf, 1);
    assert_eq!(records.bsdfs[0].sigma, 2);
    assert_eq!(records.bsdfs[0].vndf, 3);
    assert_eq!(records.bsdfs[0].luminance, 4);
    assert_eq!(records.bsdfs[0].spectra, 5);
    assert_eq!(records.bsdfs[0].isotropic, 1);
    assert_eq!(records.bsdfs[0].padding, [0; 2]);
    assert_eq!(records.bsdfs[1].isotropic, 0);

    assert_eq!(records.tables.len(), 1);
    assert_eq!(records.tables[0].size, [11, 12]);
    assert_eq!(records.tables[0].parameter_count, 2);
    assert_eq!(records.tables[0].padding0, 0);
    assert_eq!(records.tables[0].parameter_sizes, [13, 14, 15]);
    assert_eq!(records.tables[0].padding1, 0);
    assert_eq!(records.tables[0].parameter_strides, [16, 17, 18]);
    assert_eq!(records.tables[0].padding2, 0);
    assert_eq!(records.tables[0].parameter_value_offsets, [19, 20, 21]);
    assert_eq!(records.tables[0].padding3, 0);
    assert_eq!(records.tables[0].data_offset, 22);
    assert_eq!(records.tables[0].marginal_cdf_offset, 23);
    assert_eq!(records.tables[0].conditional_cdf_offset, 24);
    assert_eq!(records.tables[0].padding4, 0);
}
