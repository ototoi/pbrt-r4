use std::path::Path;

use crate::util::base::Float;
use crate::util::misc::read_float_file;

use super::blackbody::BlackbodySpectrum;
use super::composite::Spectrum;
use super::named::{lookup_named_spectrum, lookup_named_spectrum_curve};

pub const SPECTRUM_CLASS_REFLECTANCE: &str = "reflectance";
pub const SPECTRUM_CLASS_ALBEDO: &str = "albedo";
pub const SPECTRUM_CLASS_UNBOUNDED: &str = "unbounded";
pub const SPECTRUM_CLASS_ILLUMINANT: &str = "illuminant";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumType {
    Illuminant,
    Albedo,
    Unbounded,
}

impl SpectrumType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Illuminant => SPECTRUM_CLASS_ILLUMINANT,
            Self::Albedo => SPECTRUM_CLASS_ALBEDO,
            Self::Unbounded => SPECTRUM_CLASS_UNBOUNDED,
        }
    }
}

pub fn canonical_spectrum_type(spectrum_class: &str) -> Option<SpectrumType> {
    match spectrum_class {
        SPECTRUM_CLASS_REFLECTANCE | SPECTRUM_CLASS_ALBEDO => Some(SpectrumType::Albedo),
        SPECTRUM_CLASS_UNBOUNDED => Some(SpectrumType::Unbounded),
        SPECTRUM_CLASS_ILLUMINANT => Some(SpectrumType::Illuminant),
        _ => None,
    }
}

pub fn canonical_spectrum_class(spectrum_class: &str) -> Option<&'static str> {
    canonical_spectrum_type(spectrum_class).map(SpectrumType::as_str)
}

pub fn load_samples_from_file(path: &str) -> Option<Spectrum> {
    spectrum_from_file(path)
}

pub fn spectrum_from_file(path: &str) -> Option<Spectrum> {
    if !Path::new(path).exists() {
        return None;
    }
    let vals = read_float_file(path).ok()?;
    let mut lambda = Vec::with_capacity(vals.len() / 2);
    let mut values = Vec::with_capacity(vals.len() / 2);
    for chunk in vals.chunks_exact(2) {
        lambda.push(chunk[0]);
        values.push(chunk[1]);
    }
    Some(Spectrum::from_sampled(&lambda, &values))
}

pub fn spectrum_from_named(name: &str) -> Option<Spectrum> {
    if let Some((lambda, values)) = lookup_named_spectrum_curve(name) {
        return Some(Spectrum::from_sampled(&lambda, &values));
    }
    lookup_named_spectrum(name)
}

pub fn blackbody_spectrum(values: &[Float]) -> Spectrum {
    let temperature = values.first().copied().unwrap_or(0.0);
    let scale = values.get(1).copied().unwrap_or(1.0);
    Spectrum::Blackbody(BlackbodySpectrum::new(temperature, scale))
}
