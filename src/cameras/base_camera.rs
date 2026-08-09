use crate::base::camera::{Camera, CameraRay, CameraRayDifferential, CameraSample};
use crate::film::Film;
use crate::media::*;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::spectrum::SampledWavelengths;
use crate::util::transform::*;
use crate::util::vecmath::Frame;

use log::*;
use std::sync::Arc;
use std::sync::RwLock;

/// pbrt-v4 `CameraBaseParameters` -- bundles the data every Camera
/// subclass needs from the parameter dictionary at construction time
/// (animation transform, shutter window, target film, surrounding
/// medium). Concrete cameras (`PerspectiveCamera`, `OrthographicCamera`,
/// ...) take this as the first constructor argument plus their own
/// type-specific knobs.
#[derive(Clone)]
pub struct CameraBaseParameters {
    pub camera_to_world: AnimatedTransform,
    pub shutter_open: Float,
    pub shutter_close: Float,
    pub film: Arc<RwLock<Film>>,
    pub medium: Option<Arc<Medium>>,
}

impl CameraBaseParameters {
    pub fn new(
        camera_to_world: &AnimatedTransform,
        shutter_open: Float,
        shutter_close: Float,
        film: &Arc<RwLock<Film>>,
        medium: &Option<Arc<Medium>>,
    ) -> Self {
        Self {
            camera_to_world: camera_to_world.clone(),
            shutter_open,
            shutter_close,
            film: film.clone(),
            medium: medium.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CameraDifferentials {
    pub min_pos_differential_x: Vector3f,
    pub min_pos_differential_y: Vector3f,
    pub min_dir_differential_x: Vector3f,
    pub min_dir_differential_y: Vector3f,
}

impl CameraDifferentials {
    fn infinity() -> Self {
        Self {
            min_pos_differential_x: Vector3f::new(
                Float::INFINITY,
                Float::INFINITY,
                Float::INFINITY,
            ),
            min_pos_differential_y: Vector3f::new(
                Float::INFINITY,
                Float::INFINITY,
                Float::INFINITY,
            ),
            min_dir_differential_x: Vector3f::new(
                Float::INFINITY,
                Float::INFINITY,
                Float::INFINITY,
            ),
            min_dir_differential_y: Vector3f::new(
                Float::INFINITY,
                Float::INFINITY,
                Float::INFINITY,
            ),
        }
    }

    fn is_finite(&self) -> bool {
        vector3_is_finite(&self.min_pos_differential_x)
            && vector3_is_finite(&self.min_pos_differential_y)
            && vector3_is_finite(&self.min_dir_differential_x)
            && vector3_is_finite(&self.min_dir_differential_y)
    }
}

impl Default for CameraDifferentials {
    fn default() -> Self {
        Self::infinity()
    }
}

fn vector3_is_finite(v: &Vector3f) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[derive(Clone)]
pub struct BaseCamera {
    pub camera_to_world: AnimatedTransform,
    pub shutter_open: Float,
    pub shutter_close: Float,
    pub film: Arc<RwLock<Film>>,
    pub medium: Option<Arc<Medium>>,
    pub differentials: CameraDifferentials,
}

impl BaseCamera {
    pub fn new(params: CameraBaseParameters) -> Self {
        if params.camera_to_world.has_scale() {
            warn!(
                "Scaling detected in world-to-camera transformation!\nThe system has numerous assumptions, implicit and explicit,\nthat this transform will have no scale factors in it.\nProceed at your own risk; your image may have errors or\nthe system may crash as a result of this."
            );
        }

        BaseCamera {
            camera_to_world: params.camera_to_world,
            shutter_open: params.shutter_open,
            shutter_close: params.shutter_close,
            film: params.film,
            medium: params.medium,
            differentials: CameraDifferentials::default(),
        }
    }

    pub fn get_film(&self) -> Arc<RwLock<Film>> {
        return Arc::clone(&self.film);
    }

    pub fn get_medium(&self) -> Option<Arc<Medium>> {
        return self.medium.clone();
    }

    pub fn get_shutter(&self) -> (Float, Float) {
        return (self.shutter_open, self.shutter_close);
    }

    pub fn get_camera_to_world(&self) -> AnimatedTransform {
        self.camera_to_world.clone()
    }

    pub fn set_minimum_differentials(&mut self, differentials: CameraDifferentials) {
        self.differentials = differentials;
    }

    pub fn find_minimum_differentials(camera: &Camera) -> CameraDifferentials {
        let mut differentials = CameraDifferentials::infinity();

        let film = camera.get_film();
        let full_resolution = film.read().unwrap().full_resolution();
        let mut sample = CameraSample {
            p_lens: Point2f::new(0.5, 0.5),
            time: 0.5,
            ..Default::default()
        };
        let lambda = SampledWavelengths::sample_visible(0.5);

        let n = 512;
        for i in 0..n {
            let t = i as Float / (n - 1) as Float;
            sample.p_film.x = t * full_resolution.x as Float;
            sample.p_film.y = t * full_resolution.y as Float;

            let Some(crd) = camera.generate_ray_differential(&sample, &lambda) else {
                continue;
            };
            let ray = crd.ray;
            if !ray.has_differentials
                || ray.ray.d.length_squared() == 0.0
                || ray.rx_direction.length_squared() == 0.0
                || ray.ry_direction.length_squared() == 0.0
            {
                continue;
            }

            let camera_from_render = camera
                .get_camera_to_world()
                .interpolate(ray.ray.time)
                .inverse();
            let dox = camera_from_render.transform_vector(&(ray.rx_origin - ray.ray.o));
            if dox.length() < differentials.min_pos_differential_x.length() {
                differentials.min_pos_differential_x = dox;
            }
            let doy = camera_from_render.transform_vector(&(ray.ry_origin - ray.ray.o));
            if doy.length() < differentials.min_pos_differential_y.length() {
                differentials.min_pos_differential_y = doy;
            }

            let d = ray.ray.d.normalize();
            let rx_direction = ray.rx_direction.normalize();
            let ry_direction = ray.ry_direction.normalize();
            let f = Frame::from_z(d);
            let df = f.to_local(d);
            let dxf = f.to_local(rx_direction).normalize();
            let dyf = f.to_local(ry_direction).normalize();

            let ddx = dxf - df;
            if ddx.length() < differentials.min_dir_differential_x.length() {
                differentials.min_dir_differential_x = ddx;
            }
            let ddy = dyf - df;
            if ddy.length() < differentials.min_dir_differential_y.length() {
                differentials.min_dir_differential_y = ddy;
            }
        }

        differentials
    }

    pub fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: u32,
    ) -> Option<(Vector3f, Vector3f)> {
        if !self.differentials.is_finite() || p.length_squared() == 0.0 {
            return None;
        }

        let camera_from_render = self.camera_to_world.interpolate(time).inverse();
        let render_from_camera = self.camera_to_world.interpolate(time);
        let p_camera = camera_from_render.transform_point(&p);
        if p_camera.length_squared() == 0.0 {
            return None;
        }

        let down_z_from_camera =
            Transform::rotate_from_to(p_camera.normalize(), Vector3f::new(0.0, 0.0, 1.0));
        let p_down_z = down_z_from_camera.transform_point(&p_camera);
        let n_camera = camera_from_render.transform_normal(&n);
        let n_down_z = down_z_from_camera.transform_normal(&n_camera);
        let d = n_down_z.z * p_down_z.z;

        let x_ray_origin = self.differentials.min_pos_differential_x;
        let x_ray_direction =
            Vector3f::new(0.0, 0.0, 1.0) + self.differentials.min_dir_differential_x;
        let denom_x = Vector3f::dot(&n_down_z, &x_ray_direction);
        if denom_x == 0.0 {
            return None;
        }
        let tx = -(Vector3f::dot(&n_down_z, &x_ray_origin) - d) / denom_x;

        let y_ray_origin = self.differentials.min_pos_differential_y;
        let y_ray_direction =
            Vector3f::new(0.0, 0.0, 1.0) + self.differentials.min_dir_differential_y;
        let denom_y = Vector3f::dot(&n_down_z, &y_ray_direction);
        if denom_y == 0.0 {
            return None;
        }
        let ty = -(Vector3f::dot(&n_down_z, &y_ray_origin) - d) / denom_y;

        if !tx.is_finite() || !ty.is_finite() {
            return None;
        }

        let px = x_ray_origin + tx * x_ray_direction;
        let py = y_ray_origin + ty * y_ray_direction;
        let spp_scale = Float::max(0.125, 1.0 / Float::sqrt(samples_per_pixel.max(1) as Float));
        let inv_down_z = down_z_from_camera.inverse();
        let dpdx = spp_scale
            * render_from_camera.transform_vector(&inv_down_z.transform_vector(&(px - p_down_z)));
        let dpdy = spp_scale
            * render_from_camera.transform_vector(&inv_down_z.transform_vector(&(py - p_down_z)));

        if vector3_is_finite(&dpdx) && vector3_is_finite(&dpdy) {
            Some((dpdx, dpdy))
        } else {
            None
        }
    }

    //ray diff helper
    pub fn generate_ray_differential(
        camera: &Camera,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        let CameraRay {
            ray: rd,
            weight: wt,
        } = camera.generate_ray(sample, lambda)?;

        let mut rx_origin = Vector3f::zero();
        let mut rx_direction = Vector3f::zero();
        let mut wtx = 0.0;
        for eps in [0.05, -0.05] {
            let mut sshift = *sample;
            sshift.p_film.x += eps;
            if let Some(rx) = camera.generate_ray(&sshift, lambda) {
                wtx = rx.weight;
                rx_origin = rd.o + (rx.ray.o - rd.o) / eps;
                rx_direction = rd.d + (rx.ray.d - rd.d) / eps;
                break;
            }
        }
        if wtx == 0.0 {
            return None;
        }

        let mut ry_origin = Vector3f::zero();
        let mut ry_direction = Vector3f::zero();
        let mut wty = 0.0;
        for eps in [0.05, -0.05] {
            let mut sshift = *sample;
            sshift.p_film.y += eps;
            if let Some(ry) = camera.generate_ray(&sshift, lambda) {
                wty = ry.weight;
                ry_origin = rd.o + (ry.ray.o - rd.o) / eps;
                ry_direction = rd.d + (ry.ray.d - rd.d) / eps;
                break;
            }
        }
        if wty == 0.0 {
            return None;
        }
        let rd = RayDifferential {
            ray: rd,
            has_differentials: true,
            rx_origin,
            rx_direction,
            ry_origin,
            ry_direction,
        };

        return Some(CameraRayDifferential {
            ray: rd,
            weight: wt,
        });
    }
}
