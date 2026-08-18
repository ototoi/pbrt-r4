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
use crate::util::sampling::*;
use crate::util::spectrum::*;
use crate::util::transform::*;

use log::*;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone)]
pub struct OrthographicCamera {
    base: ProjectiveCamera,
    dx_camera: Vector3f,
    dy_camera: Vector3f,
    a: Float,
}

impl OrthographicCamera {
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
        let lensradius = params.get_one_float("lensradius", 0.0);
        let focaldistance = params.get_one_float("focaldistance", 1e6);

        let frame = {
            let film = film.read().unwrap();
            let full_resolution = film.full_resolution();
            full_resolution.x as Float / full_resolution.y as Float
        };
        let frame = params.get_one_float("frameaspectratio", frame);
        let mut screen = if frame > 1.0 {
            Bounds2f {
                min: Point2f { x: -frame, y: -1.0 },
                max: Point2f { x: frame, y: 1.0 },
            }
        } else {
            Bounds2f {
                min: Point2f {
                    x: -1.0,
                    y: -1.0 / frame,
                },
                max: Point2f {
                    x: 1.0,
                    y: 1.0 / frame,
                },
            }
        };
        if let Some(sw) = params.get_floats_ref("screenwindow") {
            if sw.len() == 4 {
                screen.min.x = sw[0];
                screen.max.x = sw[1];
                screen.min.y = sw[2];
                screen.max.y = sw[3];
            } else {
                error!("Screen window should have four values. Using default.");
            }
        }

        let base_params =
            CameraBaseParameters::new(cam2world, shutteropen, shutterclose, film, medium);
        Ok(Self::new(base_params, &screen, lensradius, focaldistance))
    }

    pub fn new(
        base_params: CameraBaseParameters,
        screen_window: &Bounds2f,
        lens_radius: Float,
        focal_distance: Float,
    ) -> Self {
        let camera_to_screen = Transform::orthographic(0.0, 1.0);
        let base = ProjectiveCamera::new(
            base_params,
            &camera_to_screen,
            screen_window,
            lens_radius,
            focal_distance,
        );
        let dx_camera = base
            .raster_to_camera
            .transform_vector(&Vector3f::new(1.0, 0.0, 0.0));
        let dy_camera = base
            .raster_to_camera
            .transform_vector(&Vector3f::new(0.0, 1.0, 0.0));
        let full_resolution = base.base.film.read().unwrap().full_resolution();
        let min = base
            .raster_to_camera
            .transform_point(&Point3f::new(0.0, 0.0, 0.0));
        let max = base.raster_to_camera.transform_point(&Point3f::new(
            full_resolution.x as Float,
            full_resolution.y as Float,
            0.0,
        ));
        let a = Float::abs((max.x - min.x) * (max.y - min.y));
        OrthographicCamera {
            base,
            dx_camera,
            dy_camera,
            a,
        }
    }

    pub fn get_camera_to_world(&self) -> AnimatedTransform {
        self.base.get_camera_to_world()
    }

    pub fn init_minimum_differentials(&mut self) {
        let differentials =
            BaseCamera::find_minimum_differentials(&Camera::Orthographic(self.clone()));
        self.base.base.set_minimum_differentials(differentials);
    }

    pub fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: u32,
    ) -> Option<(Vector3f, Vector3f)> {
        self.base
            .base
            .approximate_dp_dxy(p, n, time, samples_per_pixel)
    }
}

impl OrthographicCamera {
    pub fn generate_ray(
        &self,
        sample: &CameraSample,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraRay> {
        // Compute raster and camera sample positions
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = self.base.raster_to_camera.transform_point(&p_film);
        let mut ray = Ray::new(
            &p_camera,
            &Vector3f::new(0.0, 0.0, 1.0),
            Float::INFINITY,
            sample.time,
        );
        // Modify ray for depth of field
        if self.base.lens_radius > 0.0 {
            // Sample point on lens
            let p_lens = self.base.lens_radius * concentric_sample_disk(&sample.p_lens);

            // Compute point on plane of focus
            let ft = self.base.focal_distance / ray.d.z;
            let p_focus = ray.position(ft);

            // Update ray for effect of lens
            ray.o = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.d = (p_focus - ray.o).normalize();
        }
        ray.time = lerp(
            sample.time,
            self.base.base.shutter_open,
            self.base.base.shutter_close,
        );
        ray.medium = self.base.base.get_medium();
        let (ray, _, _) = self.base.base.camera_to_world.transform_ray(&ray);
        return Some(CameraRay { ray, weight: 1.0 });
    }

    pub fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = self.base.raster_to_camera.transform_point(&p_film);

        let mut ray = RayDifferential::new(
            &p_camera,
            &Vector3f::new(0.0, 0.0, 1.0),
            Float::INFINITY,
            sample.time,
        );
        // Modify ray for depth of field
        if self.base.lens_radius > 0.0 {
            // Sample point on lens
            let p_lens = concentric_sample_disk(&sample.p_lens) * self.base.lens_radius;

            // Compute point on plane of focus
            let ft = self.base.focal_distance / ray.ray.d.z;
            let p_focus = ray.ray.position(ft);

            // Update ray for effect of lens
            ray.ray.o = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.ray.d = (p_focus - ray.ray.o).normalize();
        }

        // Compute offset rays for _PerspectiveCamera_ ray differentials
        if self.base.lens_radius > 0.0 {
            // Compute _PerspectiveCamera_ ray differentials accounting for lens

            // Sample point on lens
            let p_lens = concentric_sample_disk(&sample.p_lens) * self.base.lens_radius;
            {
                let dx = (p_camera + self.dx_camera).normalize();
                let ft = self.base.focal_distance / dx.z;
                let p_focus = p_camera + self.dx_camera + (ft * Vector3f::new(0.0, 0.0, 1.0));
                ray.rx_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
                ray.rx_direction = (p_focus - ray.rx_origin).normalize();
            }
            {
                let dy = (p_camera + self.dy_camera).normalize();
                let ft = self.base.focal_distance / dy.z;
                let p_focus = p_camera + self.dy_camera + (ft * Vector3f::new(0.0, 0.0, 1.0));
                ray.ry_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
                ray.ry_direction = (p_focus - ray.ry_origin).normalize();
            }
        } else {
            ray.rx_origin = ray.ray.o + self.dx_camera;
            ray.ry_origin = ray.ray.o + self.dy_camera;
            ray.rx_direction = ray.ray.d;
            ray.ry_direction = ray.ray.d;
        }

        ray.ray.time = lerp(
            sample.time,
            self.base.base.shutter_open,
            self.base.base.shutter_close,
        );
        ray.ray.medium = self.base.base.get_medium();

        let (mut ray, _, _) = self
            .base
            .base
            .camera_to_world
            .transform_ray_differential(&ray);
        ray.has_differentials = true;

        return Some(CameraRayDifferential { ray, weight: 1.0 });
    }

    pub fn we(&self, ray: &Ray) -> Option<(Spectrum, Point2f)> {
        if self.base.lens_radius > 0.0 {
            return None;
        }

        let c2w = self.base.base.camera_to_world.interpolate(ray.time);
        let w2c = c2w.inverse();
        let z = c2w.transform_vector(&Vector3f::new(0.0, 0.0, 1.0));
        if ray.d.dot(&z) <= 1.0 - 1e-5 {
            return None;
        }

        let p_camera = w2c.transform_point(&ray.o);
        let camera_to_raster = self.base.raster_to_camera.inverse();
        let p_raster = camera_to_raster.transform_point(&p_camera);
        let p_raster = Point2f::new(p_raster.x, p_raster.y);

        {
            let film = self.get_film();
            let film = film.read().unwrap();
            let sample_bounds = film.sample_bounds();
            if p_raster.x < sample_bounds.min.x as Float
                || p_raster.x >= sample_bounds.max.x as Float
                || p_raster.y < sample_bounds.min.y as Float
                || p_raster.y >= sample_bounds.max.y as Float
            {
                return None;
            }
        }

        Some((Spectrum::from(1.0 / self.a), p_raster))
    }

    pub fn pdf_we(&self, ray: &Ray) -> Option<(Float, Float)> {
        if self.base.lens_radius > 0.0 {
            return None;
        }

        let c2w = self.base.base.camera_to_world.interpolate(ray.time);
        let w2c = c2w.inverse();
        let z = c2w.transform_vector(&Vector3f::new(0.0, 0.0, 1.0));
        if ray.d.dot(&z) <= 1.0 - 1e-5 {
            return None;
        }

        let p_camera = w2c.transform_point(&ray.o);
        let camera_to_raster = self.base.raster_to_camera.inverse();
        let p_raster = camera_to_raster.transform_point(&p_camera);

        {
            let film = self.get_film();
            let film = film.read().unwrap();
            let sample_bounds = film.sample_bounds();
            if p_raster.x < sample_bounds.min.x as Float
                || p_raster.x >= sample_bounds.max.x as Float
                || p_raster.y < sample_bounds.min.y as Float
                || p_raster.y >= sample_bounds.max.y as Float
            {
                return None;
            }
        }

        Some((1.0 / self.a, 1.0))
    }

    pub fn sample_wi(
        &self,
        inter: &Interaction,
        _u: &Point2f,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraWiSample> {
        if self.base.lens_radius > 0.0 {
            return None;
        }

        let time = inter.get_time();
        let c2w = self.base.base.camera_to_world.interpolate(time);
        let w2c = c2w.inverse();
        let z_world = c2w.transform_vector(&Vector3f::new(0.0, 0.0, 1.0));

        let p_ref_camera = w2c.transform_point(&inter.get_p());
        let p_camera = Point3f::new(p_ref_camera.x, p_ref_camera.y, 0.0);
        let p_world = c2w.transform_point(&p_camera);

        let wi = p_world - inter.get_p();
        let dist2 = wi.length_squared();
        if dist2 == 0.0 {
            return None;
        }
        let wi = wi / Float::sqrt(dist2);
        if wi.dot(&-z_world) <= 1.0 - 1e-5 {
            return None;
        }

        let camera_to_raster = self.base.raster_to_camera.inverse();
        let p_raster = camera_to_raster.transform_point(&p_camera);
        let p_raster = Point2f::new(p_raster.x, p_raster.y);
        {
            let film = self.get_film();
            let film = film.read().unwrap();
            let sample_bounds = film.sample_bounds();
            if p_raster.x < sample_bounds.min.x as Float
                || p_raster.x >= sample_bounds.max.x as Float
                || p_raster.y < sample_bounds.min.y as Float
                || p_raster.y >= sample_bounds.max.y as Float
            {
                return None;
            }
        }

        let medium = self.base.base.medium.clone();
        let lens_intr = Interaction::Base(BaseInteraction {
            p: p_world,
            time,
            medium_interface: MediumInterface::from(&medium),
            n: Normal3f::from(z_world),
            ..Default::default()
        });
        let vis = VisibilityTester::from((inter.clone(), lens_intr));
        let spec = Spectrum::from(1.0 / self.a);
        Some(CameraWiSample {
            wi_spec: spec,
            wi,
            pdf: 1.0,
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
