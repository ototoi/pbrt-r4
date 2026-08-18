use super::majorant_iterator::DDAMajorantIterator;
use super::majorant_iterator::MajorantGrid;
use super::medium::grid_resolution;
use super::phase_function::HGPhaseFunction;
use super::phase_function::PhaseFunction;
use super::sample_grid::SampledGrid;
use crate::base::{MediumCoefficients, MediumProperties, MediumSigma};
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;

use crate::util::base::*;
use crate::util::geometry::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;
use crate::util::transform::*;

use std::sync::Arc;

type FloatGrid = SampledGrid<Float>;

#[derive(Debug, Clone)]
pub struct GridMedium {
    bounds: Bounds3f,
    render_from_medium: Transform,
    sigma_a_spec: Spectrum,
    sigma_s_spec: Spectrum,
    density_grid: FloatGrid,
    phase: Arc<PhaseFunction>,
    temperature_grid: Option<FloatGrid>,
    le_spec: Spectrum,
    le_scale: FloatGrid,
    is_emissive: bool,
    temperature_scale: Float,
    temperature_offset: Float,
    majorant_grid: MajorantGrid,
}

impl GridMedium {
    pub fn create(
        parameters: &ParameterDictionary,
        render_from_medium: &Transform,
    ) -> Result<Self, PbrtError> {
        let (nx, ny, nz) = grid_resolution(parameters);
        let expected = (nx * ny * nz) as usize;
        let Some(density) = parameters.get_floats_ref("density") else {
            return Err(PbrtError::error(
                "GridMedium requested without density grid.",
            ));
        };
        if density.len() != expected {
            return Err(PbrtError::error(&format!(
                "GridMedium has density grid length {}, expected {}.",
                density.len(),
                expected
            )));
        }

        let temperature = parameters.get_floats("temperature");
        if !temperature.is_empty() && temperature.len() != expected {
            return Err(PbrtError::error(&format!(
                "GridMedium has temperature grid length {}, expected {}.",
                temperature.len(),
                expected
            )));
        }

        let le_default = Spectrum::zero();
        let mut le = parameters.get_one_spectrum_typed("Le", &le_default, SpectrumType::Illuminant);
        if !temperature.is_empty() && !le.is_black() {
            return Err(PbrtError::error(
                "GridMedium cannot specify both Le and temperature.",
            ));
        }
        let le_norm = if le.is_black() {
            le = Spectrum::zero();
            1.0
        } else {
            1.0 / spectrum_to_photometric(&le)
        };
        let le_scale_values = parameters.get_floats("Lescale");
        let le_scale = if le_scale_values.is_empty() {
            FloatGrid::new(1, 1, 1, vec![le_norm])
        } else if le_scale_values.len() == expected {
            let values = le_scale_values
                .iter()
                .map(|value| *value * le_norm)
                .collect::<Vec<_>>();
            FloatGrid::new(nx, ny, nz, values)
        } else {
            return Err(PbrtError::error(&format!(
                "GridMedium has Lescale grid length {}, expected {}.",
                le_scale_values.len(),
                expected
            )));
        };

        let sigma_a =
            parameters.get_one_spectrum_typed("sigma_a", &Spectrum::one(), SpectrumType::Unbounded);
        let sigma_s =
            parameters.get_one_spectrum_typed("sigma_s", &Spectrum::one(), SpectrumType::Unbounded);
        let scale = parameters.get_one_float("scale", 1.0);
        let g = parameters.get_one_float("g", 0.0);
        let temperature_offset = parameters.get_one_float(
            "temperatureoffset",
            parameters.get_one_float("temperaturecutoff", 0.0),
        );
        let temperature_scale = parameters.get_one_float("temperaturescale", 1.0);
        let p0 = parameters.get_one_point3f("p0", &Point3f::new(0.0, 0.0, 0.0));
        let p1 = parameters.get_one_point3f("p1", &Point3f::new(1.0, 1.0, 1.0));

        let density_grid = FloatGrid::new(nx, ny, nz, density.to_vec());
        let temperature_grid = if temperature.is_empty() {
            None
        } else {
            Some(FloatGrid::new(nx, ny, nz, temperature))
        };

        Ok(Self::new(
            &Bounds3f::new(&p0, &p1),
            render_from_medium,
            &sigma_a,
            &sigma_s,
            scale,
            g,
            density_grid,
            temperature_grid,
            temperature_scale,
            temperature_offset,
            &le,
            le_scale,
        ))
    }

    fn new(
        bounds: &Bounds3f,
        render_from_medium: &Transform,
        sigma_a: &Spectrum,
        sigma_s: &Spectrum,
        sigma_scale: Float,
        g: Float,
        density_grid: FloatGrid,
        temperature_grid: Option<FloatGrid>,
        temperature_scale: Float,
        temperature_offset: Float,
        le: &Spectrum,
        le_scale: FloatGrid,
    ) -> Self {
        let sigma_a_spec = sigma_a.clone() * sigma_scale;
        let sigma_s_spec = sigma_s.clone() * sigma_scale;
        let is_emissive = temperature_grid.is_some() || !le.is_black();
        let majorant_grid = Self::build_majorant_grid(bounds, &density_grid);

        GridMedium {
            bounds: *bounds,
            render_from_medium: *render_from_medium,
            sigma_a_spec,
            sigma_s_spec,
            density_grid,
            phase: Arc::new(PhaseFunction::from(HGPhaseFunction::new(g))),
            temperature_grid,
            le_spec: le.clone(),
            le_scale,
            is_emissive,
            temperature_scale,
            temperature_offset,
            majorant_grid,
        }
    }

    fn density_at(&self, p: &Point3f) -> Float {
        self.density_grid.lookup(p, None)
    }

    fn le_at(&self, p: &Point3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        if !self.is_emissive {
            return SampledSpectrum::zero();
        }
        let le_scale = self.le_scale.lookup(p, None);
        if le_scale <= 0.0 {
            return SampledSpectrum::zero();
        }
        if let Some(temperature_grid) = &self.temperature_grid {
            let temperature = (temperature_grid.lookup(p, None) - self.temperature_offset)
                * self.temperature_scale;
            if temperature > 100.0 {
                return blackbody_spectrum(&[temperature]).sample(lambda) * le_scale;
            }
            SampledSpectrum::zero()
        } else {
            self.le_spec.sample(lambda) * le_scale
        }
    }

    fn build_majorant_grid(bounds: &Bounds3f, density: &FloatGrid) -> MajorantGrid {
        let res: [i32; 3] = [16, 16, 16];
        let mut values = vec![0.0; (res[0] * res[1] * res[2]) as usize];
        for z in 0..res[2] {
            for y in 0..res[1] {
                for x in 0..res[0] {
                    let p0 = Point3f::new(
                        x as Float / res[0] as Float,
                        y as Float / res[1] as Float,
                        z as Float / res[2] as Float,
                    );
                    let p1 = Point3f::new(
                        (x + 1) as Float / res[0] as Float,
                        (y + 1) as Float / res[1] as Float,
                        (z + 1) as Float / res[2] as Float,
                    );
                    let bounds = Bounds3f::new(&p0, &p1);
                    values[((z * res[1] + y) * res[0] + x) as usize] = density.max_value(&bounds);
                }
            }
        }
        MajorantGrid::new(*bounds, res, Arc::from(values))
    }
}

impl GridMedium {
    pub fn is_emissive(&self) -> bool {
        self.is_emissive
    }

    pub fn sample_point(&self, p: &Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        let p_medium = Transform::inverse(&self.render_from_medium).transform_point(p);
        let p_medium = self.bounds.offset(&p_medium);
        let density = self.density_at(&p_medium);
        let le = self.le_at(&p_medium, lambda);
        // SampledSpectrum * Float is a 4-lane mul on stack, no heap.
        MediumProperties::new(
            self.sigma_a_spec.sample(lambda) * density,
            self.sigma_s_spec.sample(lambda) * density,
            Arc::clone(&self.phase),
            le,
        )
    }

    pub fn sample_point_coefficients(
        &self,
        p: &Point3f,
        lambda: &SampledWavelengths,
    ) -> MediumCoefficients {
        let p_medium = Transform::inverse(&self.render_from_medium).transform_point(p);
        let p_medium = self.bounds.offset(&p_medium);
        let density = self.density_at(&p_medium);
        let le = self.le_at(&p_medium, lambda);
        MediumCoefficients::new(
            self.sigma_a_spec.sample(lambda) * density,
            self.sigma_s_spec.sample(lambda) * density,
            le,
        )
    }

    pub fn sample_point_sigma(&self, p: &Point3f, lambda: &SampledWavelengths) -> MediumSigma {
        let p_medium = Transform::inverse(&self.render_from_medium).transform_point(p);
        let p_medium = self.bounds.offset(&p_medium);
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
    ) -> Option<DDAMajorantIterator> {
        let (ray_medium, _, _) = Transform::inverse(&self.render_from_medium).transform_ray(ray);
        let Some((t_min, t_max)) = self.bounds.intersect_p(&ray_medium, t_max) else {
            return None;
        };
        // Pre-sample sigma_t once for this ray's wavelengths so the DDA
        // hot loop in `majorant_segments` only needs `SampledSpectrum *
        // Float` per voxel (no `Spectrum::Add` / `Mul` enum churn).
        let sigma_t = self.sigma_a_spec.sample(lambda) + self.sigma_s_spec.sample(lambda);
        Some(DDAMajorantIterator::new(
            ray_medium,
            t_min,
            t_max,
            sigma_t,
            self.majorant_grid.clone(),
        ))
    }
}
