use crate::base::camera::{Camera, CameraRay, CameraRayDifferential, CameraSample, CameraWiSample};
use crate::cameras::*;
use crate::film::*;
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::paramdict::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::spectrum::*;
use crate::util::transform::*;

use log::*;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone, Copy)]
pub enum SphericalMapping {
    EqualArea,
    Equirectangular,
}

fn wrap_equal_area_square(mut uv: Point2f) -> Point2f {
    if uv.x < 0.0 {
        uv.x = -uv.x;
        uv.y = 1.0 - uv.y;
    } else if uv.x > 1.0 {
        uv.x = 2.0 - uv.x;
        uv.y = 1.0 - uv.y;
    }
    if uv.y < 0.0 {
        uv.x = 1.0 - uv.x;
        uv.y = -uv.y;
    } else if uv.y > 1.0 {
        uv.x = 1.0 - uv.x;
        uv.y = 2.0 - uv.y;
    }
    uv
}

#[derive(Clone)]
pub struct SphericalCamera {
    base: BaseCamera,
    mapping: SphericalMapping,
}

impl SphericalCamera {
    pub fn create(
        params: &ParameterDictionary,
        cam2world: &AnimatedTransform,
        film: &Arc<RwLock<Film>>,
        medium: &Option<Arc<Medium>>,
    ) -> Result<Self, PbrtError> {
        let mut shutteropen = params.get_one_float("shutteropen", 0.0);
        let mut shutterclose = params.get_one_float("shutterclose", 1.0);
        if shutterclose < shutteropen {
            warn!(
                "Shutter close time [{}] < shutter open [{}].  Swapping them.",
                shutterclose, shutteropen
            );
            std::mem::swap(&mut shutteropen, &mut shutterclose);
        }
        let mapping = match params.get_one_string("mapping", "equalarea").as_str() {
            "equalarea" => SphericalMapping::EqualArea,
            "equirectangular" => SphericalMapping::Equirectangular,
            name => {
                return Err(PbrtError::error(&format!(
                    "Unknown mapping for spherical camera: {}",
                    name
                )))
            }
        };
        let base_params =
            CameraBaseParameters::new(cam2world, shutteropen, shutterclose, film, medium);
        Ok(Self::new(base_params, mapping))
    }

    pub fn new(base_params: CameraBaseParameters, mapping: SphericalMapping) -> Self {
        let base = BaseCamera::new(base_params);
        SphericalCamera { base, mapping }
    }

    pub fn get_camera_to_world(&self) -> AnimatedTransform {
        self.base.get_camera_to_world()
    }

    pub fn init_minimum_differentials(&mut self) {
        let differentials =
            BaseCamera::find_minimum_differentials(&Camera::Spherical(self.clone()));
        self.base.set_minimum_differentials(differentials);
    }

    pub fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: u32,
    ) -> Option<(Vector3f, Vector3f)> {
        self.base.approximate_dp_dxy(p, n, time, samples_per_pixel)
    }
}

impl SphericalCamera {
    fn direction_to_raster(&self, dir_world: &Vector3f, time: Float) -> Option<Point2f> {
        if dir_world.length_squared() == 0.0 {
            return None;
        }
        let c2w = self.base.camera_to_world.interpolate(time);
        let w2c = c2w.inverse();
        let d = w2c.transform_vector(dir_world).normalize();
        let d = Vector3f::new(d.x, d.z, d.y);
        let uv = match self.mapping {
            SphericalMapping::Equirectangular => {
                let theta = Float::acos(Float::clamp(d.z, -1.0, 1.0));
                let mut phi = Float::atan2(d.y, d.x);
                if phi < 0.0 {
                    phi += 2.0 * PI;
                }
                Point2f::new(phi * INV_2_PI, theta * INV_PI)
            }
            SphericalMapping::EqualArea => equal_area_sphere_to_square(&d),
        };
        let film = self.base.get_film();
        let film = film.read().unwrap();
        let full_resolution = film.full_resolution();
        let p_raster = Point2f::new(
            uv.x * full_resolution.x as Float,
            uv.y * full_resolution.y as Float,
        );
        let sample_bounds = film.sample_bounds();
        if p_raster.x < sample_bounds.min.x as Float
            || p_raster.x >= sample_bounds.max.x as Float
            || p_raster.y < sample_bounds.min.y as Float
            || p_raster.y >= sample_bounds.max.y as Float
        {
            return None;
        }
        Some(p_raster)
    }

    fn direction_pdf(&self, dir_world: &Vector3f, time: Float) -> Option<Float> {
        let c2w = self.base.camera_to_world.interpolate(time);
        let w2c = c2w.inverse();
        let d = w2c.transform_vector(dir_world).normalize();
        match self.mapping {
            SphericalMapping::Equirectangular => {
                let d = Vector3f::new(d.x, d.z, d.y);
                let sin_theta = Float::sqrt(Float::max(0.0, 1.0 - d.z * d.z));
                if sin_theta == 0.0 {
                    return None;
                }
                Some(1.0 / (2.0 * PI * PI * sin_theta))
            }
            SphericalMapping::EqualArea => Some(INV_4_PI),
        }
    }

    pub fn generate_ray(
        &self,
        sample: &CameraSample,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraRay> {
        // Compute spherical camera ray direction.
        let uv = {
            let film = self.base.get_film();
            let film = film.read().unwrap();
            let full_resolution = film.full_resolution();
            Point2f::new(
                sample.p_film.x / full_resolution.x as Float,
                sample.p_film.y / full_resolution.y as Float,
            )
        };
        let dir = match self.mapping {
            SphericalMapping::Equirectangular => {
                let theta = PI * uv.y;
                let phi = 2.0 * PI * uv.x;
                Vector3f::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                )
            }
            SphericalMapping::EqualArea => equal_area_square_to_sphere(&wrap_equal_area_square(uv)),
        };
        let dir = Vector3f::new(dir.x, dir.z, dir.y);
        let mut ray = Ray::new(
            &Point3f::zero(),
            &dir,
            Float::INFINITY,
            lerp(sample.time, self.base.shutter_open, self.base.shutter_close),
        );
        ray.medium = self.base.get_medium();
        let (ray, _, _) = self.base.camera_to_world.transform_ray(&ray);
        return Some(CameraRay { ray, weight: 1.0 });
    }

    pub fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        BaseCamera::generate_ray_differential(&Camera::Spherical(self.clone()), sample, lambda)
    }

    pub fn we(&self, ray: &Ray) -> Option<(Spectrum, Point2f)> {
        let p_raster = self.direction_to_raster(&ray.d, ray.time)?;
        Some((Spectrum::one(), p_raster))
    }

    pub fn pdf_we(&self, ray: &Ray) -> Option<(Float, Float)> {
        self.direction_to_raster(&ray.d, ray.time)?;
        let pdf_dir = self.direction_pdf(&ray.d, ray.time)?;
        Some((1.0, pdf_dir))
    }

    pub fn sample_wi(
        &self,
        inter: &Interaction,
        _u: &Point2f,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraWiSample> {
        let time = inter.get_time();
        let c2w = self.base.camera_to_world.interpolate(time);
        let camera_origin = c2w.transform_point(&Point3f::zero());

        let wi = camera_origin - inter.get_p();
        let dist2 = wi.length_squared();
        if dist2 == 0.0 {
            return None;
        }
        let wi = wi / Float::sqrt(dist2);

        let medium = self.base.get_medium();
        let lens_intr = Interaction::Base(BaseInteraction {
            p: camera_origin,
            time,
            medium_interface: MediumInterface::from(&medium),
            n: Normal3f::from(c2w.transform_vector(&Vector3f::new(0.0, 1.0, 0.0))),
            ..Default::default()
        });
        let vis = VisibilityTester::from((inter.clone(), lens_intr.clone()));
        let ray = lens_intr.spawn_ray(&-wi);
        let (spec, p_raster) = self.we(&ray)?;
        Some(CameraWiSample {
            wi_spec: spec,
            wi,
            pdf: dist2,
            p_raster,
            visibility: vis,
        })
    }

    pub fn get_film(&self) -> Arc<RwLock<Film>> {
        self.base.get_film()
    }

    pub fn get_medium(&self) -> Option<Arc<Medium>> {
        self.base.get_medium()
    }

    pub fn get_shutter(&self) -> (Float, Float) {
        self.base.get_shutter()
    }
}
