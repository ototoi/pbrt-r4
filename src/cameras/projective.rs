use super::base_camera::{BaseCamera, CameraBaseParameters};
use crate::film::Film;
use crate::media::*;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::transform::*;

use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone)]
pub struct ProjectiveCamera {
    pub base: BaseCamera,
    pub camera_to_screen: Transform,
    pub raster_to_camera: Transform,
    pub screen_to_raster: Transform,
    pub raster_to_screen: Transform,
    pub lens_radius: Float,
    pub focal_distance: Float,
}

impl ProjectiveCamera {
    pub fn new(
        base_params: CameraBaseParameters,
        camera_to_screen: &Transform,
        screen_window: &Bounds2f,
        lensr: Float,
        focald: Float,
    ) -> Self {
        let resolution = base_params.film.read().unwrap().full_resolution();
        let screen_to_raster = Transform::scale(resolution.x as Float, resolution.y as Float, 1.0)
            * Transform::scale(
                1.0 / (screen_window.max.x - screen_window.min.x),
                1.0 / (screen_window.min.y - screen_window.max.y),
                1.0,
            )
            * Transform::translate(-screen_window.min.x, -screen_window.max.y, 0.0);
        let raster_to_screen = screen_to_raster.inverse();
        let raster_to_camera = camera_to_screen.inverse() * raster_to_screen;
        ProjectiveCamera {
            base: BaseCamera::new(base_params),
            camera_to_screen: camera_to_screen.clone(),
            raster_to_camera,
            screen_to_raster,
            raster_to_screen,
            lens_radius: lensr,
            focal_distance: focald,
        }
    }

    pub fn get_film(&self) -> Arc<RwLock<Film>> {
        return self.base.get_film();
    }

    pub fn get_medium(&self) -> Option<Arc<Medium>> {
        return self.base.get_medium();
    }

    pub fn get_shutter(&self) -> (Float, Float) {
        return self.base.get_shutter();
    }

    pub fn get_camera_to_world(&self) -> AnimatedTransform {
        self.base.get_camera_to_world()
    }
}
