use super::composite::Spectrum;
use super::named_arrays::named_piecewise_data;
use super::piecewise_linear::PiecewiseLinearSpectrum;
use super::sampled::SampledWavelengths;
use crate::util::base::Float;

/// pbrt-v4 `GetNamedSpectrum` (util/spectrum.cpp). The lookup is exact, as in
/// v4; an unknown spelling must not be silently resolved to another name.
pub fn lookup_named_spectrum(name: &str) -> Option<Spectrum> {
    let (data, normalize) = named_piecewise_data(name)?;
    PiecewiseLinearSpectrum::from_interleaved(data, normalize).map(Spectrum::PiecewiseLinear)
}

/// Returns the raw interleaved (lambda, value) sample buffer for a
/// named spectrum, when the caller needs the unnormalized sample list
/// itself (e.g. for sampler-driven inspection or unit tests).
pub fn lookup_named_spectrum_curve(name: &str) -> Option<(Vec<Float>, Vec<Float>)> {
    let (data, _) = named_piecewise_data(name)?;
    let mut lambda = Vec::with_capacity(data.len() / 2);
    let mut values = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        lambda.push(chunk[0]);
        values.push(chunk[1]);
    }
    Some((lambda, values))
}

pub fn eta_from_spectrum(
    eta: Spectrum,
    lambda: &SampledWavelengths,
    default_value: Float,
) -> Float {
    let sampled_eta = eta.sample_at(lambda[0]);
    if sampled_eta > 0.0 {
        return sampled_eta;
    }
    let eta_y = eta.y();
    if eta_y > 0.0 {
        eta_y
    } else {
        default_value
    }
}
