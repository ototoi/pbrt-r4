use crate::gpu::flat;
use crate::util::error::PbrtError;

use super::super::abi::DenseSpectrum;

pub fn convert_spectra(spectra: &[flat::DenseSpectrum]) -> Result<Vec<DenseSpectrum>, PbrtError> {
    flat::validate_dense_spectra(spectra)?;
    Ok(spectra
        .iter()
        .map(|spectrum| DenseSpectrum {
            samples: spectrum.samples,
            flags: spectrum.flags,
        })
        .collect())
}
