//! Deterministic diagnostic AOV integrator.

use crate::base::camera::{Camera, CameraSample};
use crate::base::sampler::Sampler;
use crate::cpu::integrators::{Integrator, IntegratorBase};
use crate::paramdict::ParameterDictionary;
use crate::scene::Scene;
use crate::util::base::{Float, Point2f};
use crate::util::error::PbrtError;
use crate::util::imageio::write_image_exr_channels;
use crate::util::spectrum::SampledWavelengths;

use std::sync::{Arc, RwLock};

#[derive(Clone, Copy)]
enum DiagnosticMode {
    THit,
    Normal,
    MaterialId,
}

pub struct DiagnosticIntegrator {
    camera: Arc<Camera>,
    aggregate: IntegratorBase,
    filename: String,
    miss: Float,
    mode: DiagnosticMode,
}

impl DiagnosticIntegrator {
    fn new(
        camera: &Arc<Camera>,
        scene: &Scene,
        filename: String,
        miss: Float,
        mode: DiagnosticMode,
    ) -> Self {
        Self {
            camera: Arc::clone(camera),
            aggregate: IntegratorBase::from_scene(scene),
            filename,
            miss,
            mode,
        }
    }
}

impl Integrator for DiagnosticIntegrator {
    fn render(&mut self) {
        let film = self.camera.get_film();
        let film = film.read().unwrap();
        let bounds = film.pixel_bounds();
        let resolution = film.full_resolution();
        drop(film);
        let pixel_count = bounds.area() as usize;
        let mut output = vec![self.miss; pixel_count];
        let mut normal_r = vec![0.0; pixel_count];
        let mut normal_g = vec![0.0; pixel_count];
        let mut normal_b = vec![0.0; pixel_count];
        let lambda = SampledWavelengths::sample_visible(0.5);
        for y in bounds.min.y..bounds.max.y {
            for x in bounds.min.x..bounds.max.x {
                let sample = CameraSample {
                    p_film: Point2f::new(x as Float + 0.5, y as Float + 0.5),
                    ..Default::default()
                };
                let index = ((y - bounds.min.y) * bounds.diagonal().x + x - bounds.min.x) as usize;
                if let Some(camera_ray) = self.camera.generate_ray(&sample, &lambda) {
                    if let Some(mut hit) =
                        self.aggregate.intersect(&camera_ray.ray, Float::INFINITY)
                    {
                        match self.mode {
                            DiagnosticMode::THit => output[index] = hit.t_hit,
                            DiagnosticMode::MaterialId => {
                                output[index] = hit
                                    .intr
                                    .material
                                    .as_ref()
                                    .map_or(self.miss, |m| m.diagnostic_id());
                            }
                            DiagnosticMode::Normal => {
                                if let Some(ray_diff) =
                                    self.camera.generate_ray_differential(&sample, &lambda)
                                {
                                    let mut normal_lambda = lambda;
                                    let _ = hit.intr.get_bsdf(
                                        &ray_diff.ray,
                                        &self.camera,
                                        1,
                                        &mut normal_lambda,
                                        None,
                                    );
                                }
                                let n = hit.intr.shading.n;
                                normal_r[index] = 0.5 * (n.x + 1.0);
                                normal_g[index] = 0.5 * (n.y + 1.0);
                                normal_b[index] = 0.5 * (n.z + 1.0);
                            }
                        }
                    }
                }
            }
        }
        let _ = match self.mode {
            DiagnosticMode::Normal => write_image_exr_channels(
                &self.filename,
                &bounds,
                &resolution,
                &[("R", normal_r), ("G", normal_g), ("B", normal_b)],
            ),
            DiagnosticMode::THit | DiagnosticMode::MaterialId => {
                write_image_exr_channels(&self.filename, &bounds, &resolution, &[("Y", output)])
            }
        };
    }

    fn get_camera(&self) -> Arc<Camera> {
        Arc::clone(&self.camera)
    }
}

pub fn create_diagnostic_integrator(
    params: &ParameterDictionary,
    _sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    let filename = params.get_one_string("filename", "depth.exr");
    let miss = params.get_one_float("miss", 0.0);
    let mode = params.get_one_string("mode", "t_hit");
    let mode = match mode.as_str() {
        "t_hit" => DiagnosticMode::THit,
        "normal" => DiagnosticMode::Normal,
        "material_id" => DiagnosticMode::MaterialId,
        _ => {
            return Err(PbrtError::error(&format!(
                "Unknown diagnostic mode \"{}\".",
                mode
            )));
        }
    };
    Ok(Arc::new(RwLock::new(DiagnosticIntegrator::new(
        camera, scene, filename, miss, mode,
    ))))
}
