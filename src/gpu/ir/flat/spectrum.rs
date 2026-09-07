use std::collections::HashMap;

use crate::util::error::PbrtError;
use crate::util::spectrum::{DenselySampledSpectrum, Spectrum};

pub const DENSE_LAMBDA_MIN: i32 = 360;
pub const DENSE_LAMBDA_MAX: i32 = 830;
pub const DENSE_SAMPLE_COUNT: usize = 471;
pub const INVALID_SPECTRUM_ID: u32 = u32::MAX;
pub const SPECTRUM_FLAG_CONSTANT: u32 = 1 << 0;

pub type SpectrumId = u32;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SpectrumKey {
    flags: u32,
    samples: Box<[u32]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseSpectrum {
    pub samples: [f32; DENSE_SAMPLE_COUNT],
    pub flags: u32,
}

pub fn validate_dense_spectra(spectra: &[DenseSpectrum]) -> Result<(), PbrtError> {
    if spectra
        .iter()
        .flat_map(|spectrum| spectrum.samples)
        .any(|sample| !sample.is_finite())
    {
        return Err(PbrtError::error(
            "Flat spectrum table contains a non-finite sample.",
        ));
    }
    Ok(())
}

pub fn evaluate_dense_spectrum(
    spectra: &[DenseSpectrum],
    id: SpectrumId,
    lambda: f32,
) -> Result<f32, PbrtError> {
    let spectrum =
        usize::try_from(id).map_err(|_| PbrtError::error("Spectrum ID does not fit usize."))?;
    let samples = spectra
        .get(spectrum)
        .ok_or_else(|| PbrtError::error("Spectrum ID is outside the dense table."))?;
    if !(DENSE_LAMBDA_MIN as f32..=DENSE_LAMBDA_MAX as f32).contains(&lambda) {
        return Ok(0.0);
    }
    let x = lambda - DENSE_LAMBDA_MIN as f32;
    let i0 = x.floor() as usize;
    let i1 = (i0 + 1).min(DENSE_SAMPLE_COUNT - 1);
    Ok(samples.samples[i0] * (1.0 - x.fract()) + samples.samples[i1] * x.fract())
}

impl DenseSpectrum {
    pub fn new(samples: [f32; DENSE_SAMPLE_COUNT], flags: u32) -> Self {
        Self { samples, flags }
    }
}

#[derive(Default)]
pub struct SpectrumTableBuilder {
    table: Vec<DenseSpectrum>,
    ids: HashMap<SpectrumKey, SpectrumId>,
}

impl SpectrumTableBuilder {
    pub fn intern(&mut self, spectrum: &Spectrum) -> Result<SpectrumId, PbrtError> {
        let flags = if spectrum.is_constant_spectrum() {
            SPECTRUM_FLAG_CONSTANT
        } else {
            0
        };
        self.intern_dense(&spectrum.to_dense(), flags)
    }

    pub fn intern_dense(
        &mut self,
        spectrum: &DenselySampledSpectrum,
        flags: u32,
    ) -> Result<SpectrumId, PbrtError> {
        let mut samples = [0.0; DENSE_SAMPLE_COUNT];
        let mut bits = Vec::with_capacity(DENSE_SAMPLE_COUNT);
        for (index, sample) in samples.iter_mut().enumerate() {
            let value = spectrum[index] as f32;
            if !value.is_finite() {
                return Err(PbrtError::error(
                    "Spectrum contains a non-finite dense sample.",
                ));
            }
            let value = if value == 0.0 { 0.0 } else { value };
            *sample = value;
            bits.push(value.to_bits());
        }
        let key = SpectrumKey {
            flags,
            samples: bits.into_boxed_slice(),
        };
        if let Some(id) = self.ids.get(&key) {
            return Ok(*id);
        }
        let id = u32::try_from(self.table.len())
            .map_err(|_| PbrtError::error("Flat spectrum table exceeds the u32 ID range."))?;
        self.table.push(DenseSpectrum::new(samples, flags));
        self.ids.insert(key, id);
        Ok(id)
    }

    pub fn finish(self) -> Vec<DenseSpectrum> {
        self.table
    }
}
