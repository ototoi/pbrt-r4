use super::majorant_iterator::HomogeneousMajorantIterator;
use super::medium::get_medium_scattering_properties;
use super::phase_function::HGPhaseFunction;
use super::phase_function::PhaseFunction;
use crate::base::{MediumCoefficients, MediumProperties, MediumSigma};
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;

use crate::util::base::*;
use crate::util::geometry::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct HomogeneousMedium {
    sigma_a_spec: DenselySampledSpectrum,
    sigma_s_spec: DenselySampledSpectrum,
    le_spec: DenselySampledSpectrum,
    phase: Arc<PhaseFunction>,
}

impl HomogeneousMedium {
    pub fn create(parameters: &ParameterDictionary) -> Result<Self, PbrtError> {
        let preset = parameters.get_one_string("preset", "");
        let (sig_a_default, sig_s_default) =
            if let Some((sigma_a, sigma_s)) = get_medium_scattering_properties(&preset) {
                (sigma_a, sigma_s)
            } else {
                if !preset.is_empty() {
                    log::warn!("Medium preset \"{}\" not found.", preset);
                }
                (Spectrum::one(), Spectrum::one())
            };
        let sigma_a =
            parameters.get_one_spectrum_typed("sigma_a", &sig_a_default, SpectrumType::Unbounded);
        let sigma_s =
            parameters.get_one_spectrum_typed("sigma_s", &sig_s_default, SpectrumType::Unbounded);
        let sigma_scale = parameters.get_one_float("scale", 1.0);

        let le_default = Spectrum::zero();
        let le = parameters.get_one_spectrum_typed("Le", &le_default, SpectrumType::Illuminant);
        let photometric = spectrum_to_photometric(&le);
        let le_scale = parameters.get_one_float("Lescale", 1.0)
            / if photometric > 0.0 { photometric } else { 1.0 };
        let g = parameters.get_one_float("g", 0.0);

        Ok(Self::new(&sigma_a, &sigma_s, sigma_scale, &le, le_scale, g))
    }

    pub fn new_with_sources(sigma_a_source: Spectrum, sigma_s_source: Spectrum, g: Float) -> Self {
        Self::new(
            &sigma_a_source,
            &sigma_s_source,
            1.0,
            &Spectrum::zero(),
            1.0,
            g,
        )
    }

    pub fn new(
        sigma_a: &Spectrum,
        sigma_s: &Spectrum,
        sigma_scale: Float,
        le: &Spectrum,
        le_scale: Float,
        g: Float,
    ) -> Self {
        let mut sigma_a_spec = DenselySampledSpectrum::from_spectrum(sigma_a);
        sigma_a_spec.scale(sigma_scale);
        let mut sigma_s_spec = DenselySampledSpectrum::from_spectrum(sigma_s);
        sigma_s_spec.scale(sigma_scale);
        let mut le_spec = DenselySampledSpectrum::from_spectrum(le);
        le_spec.scale(le_scale);

        HomogeneousMedium {
            sigma_a_spec,
            sigma_s_spec,
            le_spec,
            phase: Arc::new(PhaseFunction::from(HGPhaseFunction::new(g))),
        }
    }
}

impl HomogeneousMedium {
    pub fn is_emissive(&self) -> bool {
        self.le_spec.max_value() > 0.0
    }

    pub fn sample_point(&self, _p: &Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        MediumProperties::new(
            self.sigma_a_spec.sample(lambda),
            self.sigma_s_spec.sample(lambda),
            Arc::clone(&self.phase),
            self.le_spec.sample(lambda),
        )
    }

    pub fn sample_point_coefficients(
        &self,
        _p: &Point3f,
        lambda: &SampledWavelengths,
    ) -> MediumCoefficients {
        MediumCoefficients::new(
            self.sigma_a_spec.sample(lambda),
            self.sigma_s_spec.sample(lambda),
            self.le_spec.sample(lambda),
        )
    }

    pub fn sample_point_sigma(&self, _p: &Point3f, lambda: &SampledWavelengths) -> MediumSigma {
        MediumSigma::new(
            self.sigma_a_spec.sample(lambda),
            self.sigma_s_spec.sample(lambda),
        )
    }

    pub fn sample_ray(
        &self,
        _ray: &Ray,
        t_max: Float,
        lambda: &SampledWavelengths,
    ) -> Option<HomogeneousMajorantIterator> {
        // v4 `HomogeneousMedium::SampleRay` (media.h:1149): just emit
        // one segment that spans the whole ray segment with the
        // pre-sampled sigma_t. `SampledSpectrum` lives on the stack
        // so no heap traffic per segment.
        let sigma_t_packet = self.sigma_a_spec.sample(lambda) + self.sigma_s_spec.sample(lambda);
        Some(HomogeneousMajorantIterator::new(0.0, t_max, sigma_t_packet))
    }
}
