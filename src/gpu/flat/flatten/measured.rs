use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::super::{
    MeasuredAtlasPage, MeasuredBsdfRecord, MeasuredBsdfResources, MeasuredTableRecord,
    INVALID_MEASURED_OFFSET, MEASURED_ATLAS_HEIGHT, MEASURED_ATLAS_WIDTH,
};
use crate::bxdfs::MeasuredBxDFData;
use crate::util::error::PbrtError;
use crate::util::sampling::PiecewiseLinear2D;

#[derive(Default)]
pub struct MeasuredBsdfLibrary {
    paths: HashMap<PathBuf, u32>,
    bsdfs: Vec<MeasuredBsdfRecord>,
    tables: Vec<MeasuredTableRecord>,
    scalars: Vec<f32>,
}

impl MeasuredBsdfLibrary {
    pub fn intern(&mut self, path: &Path) -> Result<u32, PbrtError> {
        let canonical = path.canonicalize().map_err(|error| {
            PbrtError::error(&format!(
                "Unable to resolve measured BSDF \"{}\": {error}",
                path.display()
            ))
        })?;
        if let Some(index) = self.paths.get(&canonical) {
            return Ok(*index);
        }
        let filename = canonical.to_string_lossy();
        let data = MeasuredBxDFData::try_from_file(&filename)?;
        let interpolants = data.interpolants.as_ref().ok_or_else(|| {
            PbrtError::error(&format!(
                "Measured BSDF \"{}\" contains no interpolants.",
                canonical.display()
            ))
        })?;
        let ndf = self.push_table(interpolants.ndf.as_ref())?;
        let sigma = self.push_table(interpolants.sigma.as_ref())?;
        let vndf = self.push_table(interpolants.vndf.as_ref())?;
        let luminance = self.push_table(interpolants.luminance.as_ref())?;
        let spectra = self.push_table(interpolants.spectra.as_ref())?;
        let index = u32::try_from(self.bsdfs.len())
            .map_err(|_| PbrtError::error("Measured BSDF table exceeds u32."))?;
        self.bsdfs.push(MeasuredBsdfRecord {
            ndf,
            sigma,
            vndf,
            luminance,
            spectra,
            isotropic: data.isotropic,
        });
        self.paths.insert(canonical, index);
        Ok(index)
    }

    pub fn finish(mut self) -> Result<MeasuredBsdfResources, PbrtError> {
        let scalars_per_page = usize::try_from(MEASURED_ATLAS_WIDTH)
            .ok()
            .and_then(|width| {
                usize::try_from(MEASURED_ATLAS_HEIGHT)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|texels| texels.checked_mul(4))
            .ok_or_else(|| PbrtError::error("Measured BSDF atlas page size overflowed."))?;
        while self.scalars.len() % 4 != 0 {
            self.scalars.push(0.0);
        }
        let mut atlas_pages = Vec::new();
        for chunk in self.scalars.chunks(scalars_per_page) {
            let texel_count = chunk.len().div_ceil(4);
            let width = usize::try_from(MEASURED_ATLAS_WIDTH).unwrap_or(2048);
            let height = texel_count.div_ceil(width).max(1);
            let mut texels = chunk.to_vec();
            texels.resize(width * height * 4, 0.0);
            atlas_pages.push(MeasuredAtlasPage {
                resolution: [
                    MEASURED_ATLAS_WIDTH,
                    u32::try_from(height)
                        .map_err(|_| PbrtError::error("Measured BSDF atlas height exceeds u32."))?,
                ],
                texels,
            });
        }
        Ok(MeasuredBsdfResources {
            bsdfs: self.bsdfs,
            tables: self.tables,
            atlas_pages,
        })
    }

    fn push_table<const N: usize>(
        &mut self,
        table: &PiecewiseLinear2D<N>,
    ) -> Result<u32, PbrtError> {
        if N > 3 {
            return Err(PbrtError::error(
                "Measured BSDF table has more than three conditioning parameters.",
            ));
        }
        let (x_size, y_size) = table.size();
        let mut parameter_sizes = [1; 3];
        let mut parameter_strides = [0; 3];
        let mut parameter_value_offsets = [INVALID_MEASURED_OFFSET; 3];
        let values = table.parameter_values();
        for dimension in 0..N {
            parameter_sizes[dimension] = to_u32(
                table.parameter_sizes()[dimension],
                "Measured BSDF parameter size",
            )?;
            parameter_strides[dimension] = to_u32(
                table.parameter_strides()[dimension],
                "Measured BSDF parameter stride",
            )?;
            parameter_value_offsets[dimension] = self.push_scalars(values[dimension])?;
        }
        let data_offset = self.push_scalars(table.data())?;
        let marginal_cdf_offset = if table.marginal_cdf().is_empty() {
            INVALID_MEASURED_OFFSET
        } else {
            self.push_scalars(table.marginal_cdf())?
        };
        let conditional_cdf_offset = if table.conditional_cdf().is_empty() {
            INVALID_MEASURED_OFFSET
        } else {
            self.push_scalars(table.conditional_cdf())?
        };
        let index = to_u32(self.tables.len(), "Measured BSDF table count")?;
        self.tables.push(MeasuredTableRecord {
            size: [
                to_u32(x_size, "Measured BSDF table width")?,
                to_u32(y_size, "Measured BSDF table height")?,
            ],
            parameter_count: N as u32,
            parameter_sizes,
            parameter_strides,
            parameter_value_offsets,
            data_offset,
            marginal_cdf_offset,
            conditional_cdf_offset,
        });
        Ok(index)
    }

    fn push_scalars(&mut self, values: &[f32]) -> Result<u32, PbrtError> {
        let offset = to_u32(self.scalars.len(), "Measured BSDF atlas scalar offset")?;
        self.scalars.extend_from_slice(values);
        Ok(offset)
    }
}

fn to_u32(value: usize, label: &str) -> Result<u32, PbrtError> {
    u32::try_from(value).map_err(|_| PbrtError::error(&format!("{label} exceeds u32.")))
}
