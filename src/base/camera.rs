use crate::base::film::Film;
use crate::interaction::*;
use crate::lights::*;
use crate::media::*;
use crate::paramdict::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::spectrum::*;
use crate::util::transform::AnimatedTransform;

use std::sync::Arc;
use std::sync::RwLock;

pub use crate::cameras::{OrthographicCamera, PerspectiveCamera, RealisticCamera, SphericalCamera};

#[derive(Debug, Clone, Copy)]
pub struct CameraSample {
    pub p_film: Point2f,
    pub p_lens: Point2f,
    pub time: Float,
    /// pbrt-v4 `CameraSample::filterWeight` (camera.h:35). The
    /// reconstruction filter's weight at the (sub-pixel) sample point;
    /// `Film::AddSample` consumes it. Default 1.0 (set explicitly via
    /// `..Default::default()` so existing samplers don't have to
    /// supply it yet).
    pub filter_weight: Float,
}

impl Default for CameraSample {
    fn default() -> Self {
        Self {
            p_film: Point2f::default(),
            p_lens: Point2f::default(),
            time: 0.0,
            filter_weight: 1.0,
        }
    }
}

#[derive(Clone)]
pub struct CameraRay {
    pub ray: Ray,
    pub weight: Float,
}

/// pbrt-v4 `struct CameraRayDifferential`.
#[derive(Clone)]
pub struct CameraRayDifferential {
    pub ray: RayDifferential,
    pub weight: Float,
}

/// pbrt-v4 `struct CameraWiSample` -- result of sampling an importance
/// connection from a reference point. `wi_spec` is v4's `Wi`
/// (uppercase) and `wi` is the lowercase direction; r4 spells them
/// out to avoid the case-only clash.
#[derive(Clone)]
pub struct CameraWiSample {
    pub wi_spec: Spectrum,
    pub wi: Vector3f,
    pub pdf: Float,
    pub p_raster: Point2f,
    pub visibility: VisibilityTester,
}

#[derive(Clone)]
pub enum Camera {
    Perspective(PerspectiveCamera),
    Orthographic(OrthographicCamera),
    Spherical(SphericalCamera),
    Realistic(RealisticCamera),
}

impl Camera {
    /// Create a Camera from a name and ParameterDictionary (ParameterDictionary in Rust)
    /// Matches pbrt-v4's Camera::Create API
    pub fn create(
        name: &str,
        params: &ParameterDictionary,
        cam_to_world: &AnimatedTransform,
        film: &Arc<RwLock<Film>>,
        medium: &Option<Arc<Medium>>,
    ) -> Result<Camera, PbrtError> {
        let mut camera = match name {
            "perspective" => Camera::Perspective(PerspectiveCamera::create(
                params,
                cam_to_world,
                film,
                medium,
            )?),
            "orthographic" => Camera::Orthographic(OrthographicCamera::create(
                params,
                cam_to_world,
                film,
                medium,
            )?),
            "spherical" => {
                Camera::Spherical(SphericalCamera::create(params, cam_to_world, film, medium)?)
            }
            "realistic" => {
                Camera::Realistic(RealisticCamera::create(params, cam_to_world, film, medium)?)
            }
            _ => {
                let msg = format!("Camera \"{}\" unknown.", name);
                return Err(PbrtError::error(&msg));
            }
        };

        camera.init_minimum_differentials();
        Ok(camera)
    }

    fn init_minimum_differentials(&mut self) {
        match self {
            Camera::Perspective(c) => c.init_minimum_differentials(),
            Camera::Orthographic(c) => c.init_minimum_differentials(),
            Camera::Spherical(c) => c.init_minimum_differentials(),
            Camera::Realistic(c) => c.init_minimum_differentials(),
        }
    }

    pub fn generate_ray(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRay> {
        match self {
            Camera::Perspective(c) => c.generate_ray(sample, lambda),
            Camera::Orthographic(c) => c.generate_ray(sample, lambda),
            Camera::Spherical(c) => c.generate_ray(sample, lambda),
            Camera::Realistic(c) => c.generate_ray(sample, lambda),
        }
    }

    pub fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        match self {
            Camera::Perspective(c) => c.generate_ray_differential(sample, lambda),
            Camera::Orthographic(c) => c.generate_ray_differential(sample, lambda),
            Camera::Spherical(c) => c.generate_ray_differential(sample, lambda),
            Camera::Realistic(c) => c.generate_ray_differential(sample, lambda),
        }
    }

    pub fn we(&self, ray: &Ray) -> Option<(Spectrum, Point2f)> {
        match self {
            Camera::Perspective(c) => c.we(ray),
            Camera::Orthographic(c) => c.we(ray),
            Camera::Spherical(c) => c.we(ray),
            Camera::Realistic(c) => c.we(ray),
        }
    }

    pub fn pdf_we(&self, ray: &Ray) -> Option<(Float, Float)> {
        match self {
            Camera::Perspective(c) => c.pdf_we(ray),
            Camera::Orthographic(c) => c.pdf_we(ray),
            Camera::Spherical(c) => c.pdf_we(ray),
            Camera::Realistic(c) => c.pdf_we(ray),
        }
    }

    pub fn sample_wi(
        &self,
        inter: &Interaction,
        u: &Point2f,
        lambda: &SampledWavelengths,
    ) -> Option<CameraWiSample> {
        match self {
            Camera::Perspective(c) => c.sample_wi(inter, u, lambda),
            Camera::Orthographic(c) => c.sample_wi(inter, u, lambda),
            Camera::Spherical(c) => c.sample_wi(inter, u, lambda),
            Camera::Realistic(c) => c.sample_wi(inter, u, lambda),
        }
    }

    pub fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: u32,
    ) -> Option<(Vector3f, Vector3f)> {
        match self {
            Camera::Perspective(c) => c.approximate_dp_dxy(p, n, time, samples_per_pixel),
            Camera::Orthographic(c) => c.approximate_dp_dxy(p, n, time, samples_per_pixel),
            Camera::Spherical(c) => c.approximate_dp_dxy(p, n, time, samples_per_pixel),
            Camera::Realistic(c) => c.approximate_dp_dxy(p, n, time, samples_per_pixel),
        }
    }

    pub fn get_film(&self) -> Arc<RwLock<Film>> {
        match self {
            Camera::Perspective(c) => c.get_film(),
            Camera::Orthographic(c) => c.get_film(),
            Camera::Spherical(c) => c.get_film(),
            Camera::Realistic(c) => c.get_film(),
        }
    }

    pub fn get_medium(&self) -> Option<Arc<Medium>> {
        match self {
            Camera::Perspective(c) => c.get_medium(),
            Camera::Orthographic(c) => c.get_medium(),
            Camera::Spherical(c) => c.get_medium(),
            Camera::Realistic(c) => c.get_medium(),
        }
    }

    pub fn get_shutter(&self) -> (Float, Float) {
        match self {
            Camera::Perspective(c) => c.get_shutter(),
            Camera::Orthographic(c) => c.get_shutter(),
            Camera::Spherical(c) => c.get_shutter(),
            Camera::Realistic(c) => c.get_shutter(),
        }
    }

    pub fn get_camera_to_world(&self) -> AnimatedTransform {
        match self {
            Camera::Perspective(c) => c.get_camera_to_world(),
            Camera::Orthographic(c) => c.get_camera_to_world(),
            Camera::Spherical(c) => c.get_camera_to_world(),
            Camera::Realistic(c) => c.get_camera_to_world(),
        }
    }
}
