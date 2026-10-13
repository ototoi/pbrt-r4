use crate::gpu::flat::MeasuredBsdfResources;

use super::abi::{MeasuredBsdfRecord, MeasuredTableRecord};

pub struct MeasuredBsdfRecords {
    pub bsdfs: Vec<MeasuredBsdfRecord>,
    pub tables: Vec<MeasuredTableRecord>,
}

impl MeasuredBsdfRecords {
    pub fn from_flat(resources: &MeasuredBsdfResources) -> Self {
        let bsdfs = resources
            .bsdfs
            .iter()
            .map(|record| MeasuredBsdfRecord {
                ndf: record.ndf,
                sigma: record.sigma,
                vndf: record.vndf,
                luminance: record.luminance,
                spectra: record.spectra,
                isotropic: u32::from(record.isotropic),
                padding: [0; 2],
            })
            .collect();
        let tables = resources
            .tables
            .iter()
            .map(|record| MeasuredTableRecord {
                size: record.size,
                parameter_count: record.parameter_count,
                padding0: 0,
                parameter_sizes: record.parameter_sizes,
                padding1: 0,
                parameter_strides: record.parameter_strides,
                padding2: 0,
                parameter_value_offsets: record.parameter_value_offsets,
                padding3: 0,
                data_offset: record.data_offset,
                marginal_cdf_offset: record.marginal_cdf_offset,
                conditional_cdf_offset: record.conditional_cdf_offset,
                padding4: 0,
            })
            .collect();
        Self { bsdfs, tables }
    }
}
