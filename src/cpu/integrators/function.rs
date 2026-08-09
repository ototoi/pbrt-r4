//! pbrt-v4's two-dimensional function/MSE integrator.

use crate::base::camera::Camera;
use crate::base::sampler::Sampler;
use crate::cpu::integrators::Integrator;
use crate::paramdict::ParameterDictionary;
use crate::scene::Scene;
use crate::util::base::{Float, Point2f, Point2i, PI, PI_OVER_4};
use crate::util::error::PbrtError;
use crate::util::geometry::Bounds2i;
use crate::util::imageio::write_image;
use crate::util::rng::RNG;

use std::fs;
use std::sync::{Arc, RwLock};

#[derive(Clone, Copy)]
enum Function {
    Step,
    Diagonal,
    Disk,
    Checkerboard,
    RotatedCheckerboard,
    Gaussian,
}

impl Function {
    fn evaluate(self, p: Point2f) -> Float {
        match self {
            Self::Step => {
                if p.x < 0.5 {
                    2.0
                } else {
                    0.0
                }
            }
            Self::Diagonal => {
                if p.x + p.y < 1.0 {
                    2.0
                } else {
                    0.0
                }
            }
            Self::Disk => {
                let dx = p.x - 0.5;
                let dy = p.y - 0.5;
                if dx * dx + dy * dy < 0.25 {
                    1.0 / (PI * 0.25)
                } else {
                    0.0
                }
            }
            Self::Checkerboard => {
                let x = (p.x * 10.0) as i32;
                let y = (p.y * 10.0) as i32;
                if ((x & 1) ^ (y & 1)) != 0 {
                    2.0
                } else {
                    0.0
                }
            }
            Self::RotatedCheckerboard => {
                let angle = PI_OVER_4;
                let (sin, cos) = angle.sin_cos();
                let q = Point2f::new(10.0 + p.x * cos - p.y * sin, 10.0 + p.x * sin + p.y * cos);
                let value = if (((q.x as i32) * 10 & 1) ^ ((q.y as i32) * 10 & 1)) != 0 {
                    2.0
                } else {
                    0.0
                };
                value / 1.0000687
            }
            Self::Gaussian => {
                fn gaussian(x: Float, mu: Float, sigma: Float) -> Float {
                    (-((x - mu) * (x - mu)) / (2.0 * sigma * sigma)).exp()
                        / (2.0 * PI * sigma * sigma).sqrt()
                }
                let mu: Float = 0.5;
                let sigma: Float = 0.25;
                let erf = |x: Float| {
                    let sign = if x < 0.0 { -1.0 } else { 1.0 };
                    let x = x.abs();
                    let t = 1.0 / (1.0 + 0.3275911 * x);
                    let y = 1.0
                        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                            - 0.284496736)
                            * t
                            - 0.254829592)
                            * t
                            * (-x * x).exp();
                    sign * y
                };
                let integral = |x: Float| 0.5 * erf((mu - x) / (sigma * (2.0 as Float).sqrt()));
                let normalization = (integral(0.0) - integral(1.0)).powi(2);
                gaussian(p.x, mu, sigma) * gaussian(p.y, mu, sigma) / normalization
            }
        }
    }
}

fn parse_function(name: &str) -> Result<Function, PbrtError> {
    match name {
        "step" => Ok(Function::Step),
        "diagonal" => Ok(Function::Diagonal),
        "disk" => Ok(Function::Disk),
        "checkerboard" => Ok(Function::Checkerboard),
        "rotatedcheckerboard" => Ok(Function::RotatedCheckerboard),
        "gaussian" => Ok(Function::Gaussian),
        _ => Err(PbrtError::error(&format!(
            "FunctionIntegrator function \"{}\" unknown.",
            name
        ))),
    }
}

pub struct FunctionIntegrator {
    camera: Arc<Camera>,
    sampler: Arc<RwLock<Sampler>>,
    function: Function,
    output_filename: String,
    image_filename: String,
    skip_bad: bool,
}

impl FunctionIntegrator {
    fn new(
        camera: &Arc<Camera>,
        sampler: &Arc<RwLock<Sampler>>,
        function: Function,
        output_filename: String,
        image_filename: String,
        skip_bad: bool,
    ) -> Self {
        Self {
            camera: Arc::clone(camera),
            sampler: Arc::clone(sampler),
            function,
            output_filename,
            image_filename,
            skip_bad,
        }
    }

    fn pixel_bounds(&self) -> Bounds2i {
        self.camera.get_film().read().unwrap().pixel_bounds()
    }
}

impl Integrator for FunctionIntegrator {
    fn render(&mut self) {
        let bounds = self.pixel_bounds();
        if !self.image_filename.is_empty() {
            let resolution = bounds.diagonal();
            let mut rgb = vec![0.0; (resolution.x * resolution.y * 3) as usize];
            let mut rng = RNG::new();
            for y in 0..resolution.y {
                for x in 0..resolution.x {
                    let mut value = 0.0;
                    for _ in 0..256 {
                        let p = Point2f::new(
                            (x as Float + rng.uniform_float()) / resolution.x as Float,
                            (y as Float + rng.uniform_float()) / resolution.y as Float,
                        );
                        value += self.function.evaluate(p);
                    }
                    value /= 256.0;
                    let i = ((y * resolution.x + x) * 3) as usize;
                    rgb[i..i + 3].fill(value);
                }
            }
            if let Err(err) = write_image(
                &self.image_filename,
                &rgb,
                &Bounds2i::from(((0, 0), (resolution.x, resolution.y))),
                &resolution,
            ) {
                log::error!("Could not write FunctionIntegrator image: {}", err);
            }
        }
        let mut sampler = self.sampler.write().unwrap();
        let spp = sampler.samples_per_pixel();
        let mut sums = vec![0.0_f64; (bounds.diagonal().x * bounds.diagonal().y) as usize];
        let mut results = String::new();
        let mut taken = 0_u32;

        for sample_index in 0..spp {
            let report = !self.skip_bad || (sample_index + 1).is_power_of_two();
            taken += 1;
            for y in bounds.min.y..bounds.max.y {
                for x in bounds.min.x..bounds.max.x {
                    let pixel = Point2i::new(x, y);
                    sampler.start_pixel_sample(pixel, sample_index, 0);
                    let u = sampler.get_pixel_2d();
                    let index =
                        ((y - bounds.min.y) * bounds.diagonal().x + x - bounds.min.x) as usize;
                    sums[index] += f64::from(self.function.evaluate(u));
                }
            }
            if report {
                let mse = sums
                    .iter()
                    .map(|v| {
                        let e = *v / f64::from(taken) - 1.0;
                        e * e
                    })
                    .sum::<f64>()
                    / sums.len() as f64;
                results.push_str(&format!("{} {mse}\n", sample_index + 1));
            }
        }

        if let Err(err) = fs::write(&self.output_filename, results) {
            log::error!(
                "Could not write FunctionIntegrator output \"{}\": {}",
                self.output_filename,
                err
            );
        }
    }

    fn get_camera(&self) -> Arc<Camera> {
        Arc::clone(&self.camera)
    }
}

pub fn create_function_integrator(
    params: &ParameterDictionary,
    sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    _scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    if matches!(*sampler.read().unwrap(), Sampler::Sobol(_)) {
        return Err(PbrtError::error(
            "\"sobol\" sampler should be replaced with \"paddedsobol\" for the \"function\" integrator.",
        ));
    }
    let name = params.get_one_string("function", "step");
    let function = parse_function(&name)?;
    let output_filename = params.get_one_string("filename", &format!("{}-mse.txt", name));
    let skip_bad = params.get_one_bool("skipbad", true);
    let image_filename = params.get_one_string("imagefilename", "");
    Ok(Arc::new(RwLock::new(FunctionIntegrator::new(
        camera,
        sampler,
        function,
        output_filename,
        image_filename,
        skip_bad,
    ))))
}
