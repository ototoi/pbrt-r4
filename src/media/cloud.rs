use super::majorant_iterator::HomogeneousMajorantIterator;
use super::medium::medium_scattering_parameters;
use super::phase_function::HGPhaseFunction;
use super::phase_function::PhaseFunction;
use crate::base::{MediumCoefficients, MediumProperties, MediumSigma};
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;

use crate::textures::*;
use crate::util::base::*;
use crate::util::geometry::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;
use crate::util::transform::*;

use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CloudMedium {
    bounds: Bounds3f,
    render_from_medium: Transform,
    phase: Arc<PhaseFunction>,
    sigma_a_spec: Spectrum,
    sigma_s_spec: Spectrum,
    density: Float,
    wispiness: Float,
    frequency: Float,
}

impl CloudMedium {
    pub fn create(
        parameters: &ParameterDictionary,
        render_from_medium: &Transform,
    ) -> Result<Self, PbrtError> {
        let density = parameters.get_one_float("density", 1.0);
        let wispiness = parameters.get_one_float("wispiness", 1.0);
        let frequency = parameters.get_one_float("frequency", 5.0);
        let (sigma_a, sigma_s, _scale, g) = medium_scattering_parameters(parameters);
        let p0 = parameters.get_one_point3f("p0", &Point3f::new(0.0, 0.0, 0.0));
        let p1 = parameters.get_one_point3f("p1", &Point3f::new(1.0, 1.0, 1.0));
        Ok(Self::new(
            &Bounds3f::new(&p0, &p1),
            render_from_medium,
            &sigma_a,
            &sigma_s,
            g,
            density,
            wispiness,
            frequency,
        ))
    }

    pub fn new(
        bounds: &Bounds3f,
        render_from_medium: &Transform,
        sigma_a: &Spectrum,
        sigma_s: &Spectrum,
        g: Float,
        density: Float,
        wispiness: Float,
        frequency: Float,
    ) -> Self {
        Self {
            bounds: *bounds,
            render_from_medium: *render_from_medium,
            phase: Arc::new(PhaseFunction::from(HGPhaseFunction::new(g))),
            sigma_a_spec: sigma_a.clone(),
            sigma_s_spec: sigma_s.clone(),
            density,
            wispiness,
            frequency,
        }
    }

    fn density_at(&self, p: &Point3f) -> Float {
        let mut pp = self.frequency * *p;
        if self.wispiness > 0.0 {
            let mut vomega = 0.05 * self.wispiness;
            let mut vlambda = 10.0;
            for _ in 0..2 {
                pp += vomega * d_noise(&(vlambda * pp));
                vomega *= 0.5;
                vlambda *= 1.99;
            }
        }

        let mut d = 0.0;
        let mut omega = 0.5;
        let mut lambda = 1.0;
        for _ in 0..5 {
            d += omega * noise(lambda * pp.x, lambda * pp.y, lambda * pp.z);
            omega *= 0.5;
            lambda *= 1.99;
        }

        let mut d = Float::clamp((1.0 - p.y) * 4.5 * self.density * d, 0.0, 1.0);
        d += 2.0 * Float::max(0.0, 0.5 - p.y);
        Float::clamp(d, 0.0, 1.0)
    }
}

impl CloudMedium {
    pub fn sample_point(&self, p: &Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        let p_medium = Transform::inverse(&self.render_from_medium).transform_point(p);
        let density = self.density_at(&p_medium);
        MediumProperties::new(
            self.sigma_a_spec.sample(lambda) * density,
            self.sigma_s_spec.sample(lambda) * density,
            Arc::clone(&self.phase),
            SampledSpectrum::zero(),
        )
    }

    pub fn sample_point_coefficients(
        &self,
        p: &Point3f,
        lambda: &SampledWavelengths,
    ) -> MediumCoefficients {
        let p_medium = Transform::inverse(&self.render_from_medium).transform_point(p);
        let density = self.density_at(&p_medium);
        MediumCoefficients::new(
            self.sigma_a_spec.sample(lambda) * density,
            self.sigma_s_spec.sample(lambda) * density,
            SampledSpectrum::zero(),
        )
    }

    pub fn sample_point_sigma(&self, p: &Point3f, lambda: &SampledWavelengths) -> MediumSigma {
        let p_medium = Transform::inverse(&self.render_from_medium).transform_point(p);
        let density = self.density_at(&p_medium);
        MediumSigma::new(
            self.sigma_a_spec.sample(lambda) * density,
            self.sigma_s_spec.sample(lambda) * density,
        )
    }

    pub fn sample_ray(
        &self,
        ray: &Ray,
        t_max: Float,
        lambda: &SampledWavelengths,
    ) -> Option<HomogeneousMajorantIterator> {
        let (ray_medium, _, _) = Transform::inverse(&self.render_from_medium).transform_ray(ray);
        let Some((t_min, t_max)) = self.bounds.intersect_p(&ray_medium, t_max) else {
            return None;
        };
        let sigma_t = self.sigma_a_spec.sample(lambda) + self.sigma_s_spec.sample(lambda);
        Some(HomogeneousMajorantIterator::new(t_min, t_max, sigma_t))
    }
}
