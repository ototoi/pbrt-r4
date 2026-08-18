use super::majorant_iterator::DDAMajorantIterator;
use super::majorant_iterator::MajorantGrid;
use super::medium::grid_resolution;
use super::phase_function::HGPhaseFunction;
use super::phase_function::PhaseFunction;
use super::sample_grid::SampledGrid;
use crate::base::{MediumCoefficients, MediumProperties, MediumSigma};
use crate::paramdict::ParameterDictionary;
use crate::util::base::*;
use crate::util::error::PbrtError;
use crate::util::geometry::*;
use crate::util::spectrum::*;
use crate::util::transform::*;

use std::sync::Arc;

/// Counterpart to pbrt-v4 `RGBGridMedium`.
#[derive(Debug, Clone)]
pub struct RGBGridMedium {
    bounds: Bounds3f,
    render_from_medium: Transform,
    le_grid: Option<SampledGrid<RGBIlluminantSpectrum>>,
    le_scale: Float,
    phase: Arc<PhaseFunction>,
    sigma_a_grid: Option<SampledGrid<RGBUnboundedSpectrum>>,
    sigma_s_grid: Option<SampledGrid<RGBUnboundedSpectrum>>,
    sigma_scale: Float,
    majorant_grid: MajorantGrid,
}

impl RGBGridMedium {
    pub fn create(
        parameters: &ParameterDictionary,
        render_from_medium: &Transform,
    ) -> Result<Self, PbrtError> {
        let (nx, ny, nz) = grid_resolution(parameters);
        let expected = (nx * ny * nz) as usize;
        let sigma_a = parameters.get_rgb_array("sigma_a");
        let sigma_s = parameters.get_rgb_array("sigma_s");
        let le = parameters.get_rgb_array("Le");
        let p0 = parameters.get_one_point3f("p0", &Point3f::new(0.0, 0.0, 0.0));
        let p1 = parameters.get_one_point3f("p1", &Point3f::new(1.0, 1.0, 1.0));

        if sigma_a.is_empty() && sigma_s.is_empty() {
            return Err(PbrtError::error(
                "RGB grid requires \"sigma_a\" and/or \"sigma_s\" parameter values.",
            ));
        }

        let n_grid = if !sigma_a.is_empty() {
            if !sigma_s.is_empty() && sigma_a.len() != sigma_s.len() {
                return Err(PbrtError::error(&format!(
                    "Different number of samples ({} vs {}) provided for \"sigma_a\" and \"sigma_s\".",
                    sigma_a.len(),
                    sigma_s.len()
                )));
            }
            sigma_a.len()
        } else {
            sigma_s.len()
        };

        if !le.is_empty() && sigma_a.is_empty() {
            return Err(PbrtError::error(
                "RGB grid requires \"sigma_a\" if \"Le\" value provided.",
            ));
        }
        if !le.is_empty() && n_grid != le.len() {
            return Err(PbrtError::error(&format!(
                "Expected {} values for \"Le\" parameter but were given {}.",
                n_grid,
                le.len()
            )));
        }
        if n_grid != expected {
            return Err(PbrtError::error(&format!(
                "RGB grid medium has {} density values; expected nx*ny*nz = {}.",
                n_grid, expected
            )));
        }

        Ok(Self::new(
            &Bounds3f::new(&p0, &p1),
            render_from_medium,
            parameters.get_one_float("g", 0.0),
            (!sigma_a.is_empty()).then_some(SampledGrid::new(
                nx,
                ny,
                nz,
                sigma_a.into_iter().map(RGBUnboundedSpectrum::new).collect(),
            )),
            (!sigma_s.is_empty()).then_some(SampledGrid::new(
                nx,
                ny,
                nz,
                sigma_s.into_iter().map(RGBUnboundedSpectrum::new).collect(),
            )),
            parameters.get_one_float("scale", 1.0),
            (!le.is_empty()).then_some(SampledGrid::new(
                nx,
                ny,
                nz,
                le.into_iter().map(RGBIlluminantSpectrum::new).collect(),
            )),
            parameters.get_one_float("Lescale", 1.0),
        ))
    }

    fn new(
        bounds: &Bounds3f,
        render_from_medium: &Transform,
        g: Float,
        sigma_a_grid: Option<SampledGrid<RGBUnboundedSpectrum>>,
        sigma_s_grid: Option<SampledGrid<RGBUnboundedSpectrum>>,
        sigma_scale: Float,
        le_grid: Option<SampledGrid<RGBIlluminantSpectrum>>,
        le_scale: Float,
    ) -> Self {
        if le_grid.is_some() {
            debug_assert!(sigma_a_grid.is_some());
        }
        let majorant_grid =
            Self::build_majorant_grid(bounds, &sigma_a_grid, &sigma_s_grid, sigma_scale);

        RGBGridMedium {
            bounds: *bounds,
            render_from_medium: *render_from_medium,
            le_grid,
            le_scale,
            phase: Arc::new(PhaseFunction::from(HGPhaseFunction::new(g))),
            sigma_a_grid,
            sigma_s_grid,
            sigma_scale,
            majorant_grid,
        }
    }

    pub fn is_emissive(&self) -> bool {
        self.le_grid.is_some() && self.le_scale > 0.0
    }

    fn build_majorant_grid(
        bounds: &Bounds3f,
        sigma_a_grid: &Option<SampledGrid<RGBUnboundedSpectrum>>,
        sigma_s_grid: &Option<SampledGrid<RGBUnboundedSpectrum>>,
        sigma_scale: Float,
    ) -> MajorantGrid {
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
                    let max_sigma_a = sigma_a_grid
                        .as_ref()
                        .map(|grid| grid.max_value(&bounds))
                        .unwrap_or(1.0);
                    let max_sigma_s = sigma_s_grid
                        .as_ref()
                        .map(|grid| grid.max_value(&bounds))
                        .unwrap_or(1.0);
                    values[((z * res[1] + y) * res[0] + x) as usize] =
                        sigma_scale * (max_sigma_a + max_sigma_s);
                }
            }
        }
        MajorantGrid::new(*bounds, res, Arc::from(values))
    }

    pub fn sample_point(&self, p: &Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        let p_medium = Transform::inverse(&self.render_from_medium).transform_point(p);
        let p_medium = self.bounds.offset(&p_medium);
        let sigma_a = self
            .sigma_a_grid
            .as_ref()
            .map(|grid| grid.lookup(&p_medium, Some(lambda)))
            .unwrap_or(SampledSpectrum::one())
            * self.sigma_scale;
        let sigma_s = self
            .sigma_s_grid
            .as_ref()
            .map(|grid| grid.lookup(&p_medium, Some(lambda)))
            .unwrap_or(SampledSpectrum::one())
            * self.sigma_scale;
        let le = if self.le_scale > 0.0 {
            self.le_grid
                .as_ref()
                .map(|grid| grid.lookup(&p_medium, Some(lambda)) * self.le_scale)
                .unwrap_or(SampledSpectrum::zero())
        } else {
            SampledSpectrum::zero()
        };

        MediumProperties::new(sigma_a, sigma_s, Arc::clone(&self.phase), le)
    }

    pub fn sample_point_coefficients(
        &self,
        p: &Point3f,
        lambda: &SampledWavelengths,
    ) -> MediumCoefficients {
        let mp = self.sample_point(p, lambda);
        MediumCoefficients::new(mp.sigma_a, mp.sigma_s, mp.le)
    }

    pub fn sample_point_sigma(&self, p: &Point3f, lambda: &SampledWavelengths) -> MediumSigma {
        let mp = self.sample_point(p, lambda);
        MediumSigma::new(mp.sigma_a, mp.sigma_s)
    }

    pub fn sample_ray(
        &self,
        ray: &Ray,
        t_max: Float,
        _lambda: &SampledWavelengths,
    ) -> Option<DDAMajorantIterator> {
        let (ray_medium, _, _) = Transform::inverse(&self.render_from_medium).transform_ray(ray);
        let Some((t_min, t_max)) = self.bounds.intersect_p(&ray_medium, t_max) else {
            return None;
        };
        Some(DDAMajorantIterator::new(
            ray_medium,
            t_min,
            t_max,
            SampledSpectrum::one(),
            self.majorant_grid.clone(),
        ))
    }
}
