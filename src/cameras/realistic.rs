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
use crate::util::lowdiscrepancy::*;
use crate::util::misc::*;
use crate::util::profile::*;
use crate::util::sampling::*;
use crate::util::scattering::*; // Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;
use crate::util::stats::*;
use crate::util::transform::*; // For refract and other scattering functions

use log::*;

use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::RwLock;

thread_local!(static RAYS: StatPercent = StatPercent::new("Camera/Rays vignetted by lens system"));

#[derive(Debug, Clone, Copy)]
pub struct LensElementInterface {
    curvature_radius: Float,
    thickness: Float,
    eta: Float,
    aperture_radius: Float,
}

#[derive(Clone)]
struct BaseRealisticCamera {
    pub base: BaseCamera,
    pub element_interfaces: Vec<LensElementInterface>,
}

impl BaseRealisticCamera {
    fn lens_rear_z(&self) -> Float {
        return self.element_interfaces.last().unwrap().thickness;
    }

    fn lens_front_z(&self) -> Float {
        return self.element_interfaces.iter().map(|e| e.thickness).sum();
    }

    fn rear_element_radius(&self) -> Float {
        return self.element_interfaces.last().unwrap().aperture_radius;
    }

    pub fn focus_binary_search(&self, focus_distance: Float) -> Float {
        let film_distance = self.focus_thick_lens(focus_distance);
        let mut film_distance_lower = film_distance;
        let mut film_distance_upper = film_distance;
        {
            while self.focus_distance(film_distance_lower) > focus_distance {
                film_distance_lower *= 1.005;
            }
        }
        {
            while self.focus_distance(film_distance_upper) < focus_distance {
                film_distance_upper /= 1.005;
            }
        }

        // Do binary search on film distances to focus
        for _ in 0..20 {
            let fmid = 0.5 * (film_distance_lower + film_distance_upper);
            let mid_focus = self.focus_distance(fmid);
            if mid_focus < focus_distance {
                film_distance_lower = fmid;
            } else {
                film_distance_upper = fmid;
            }
        }
        return 0.5 * (film_distance_lower + film_distance_upper);
    }

    pub fn focus_distance(&self, film_distance: Float) -> Float {
        // Find offset ray from film center through lens
        let film = self.base.film.as_ref().read().unwrap();
        let bounds = self.bound_exit_pupil(0.0, 0.001 * film.diagonal());

        // Try some different and decreasing scaling factor to find focus ray
        // more quickly when `aperturediameter` is too small.
        // (e.g. 2 [mm] for `aperturediameter` with wide.22mm.dat),
        let mut found_ray: Option<Ray> = None;
        for scale in [0.1, 0.01, 0.001] {
            let lu = scale * bounds.max[0];
            let ro = Point3f::new(0.0, 0.0, self.lens_rear_z() - film_distance);
            let rd = Vector3f::new(lu, 0.0, film_distance);
            let r_camera = Ray::new(&ro, &rd, Float::INFINITY, 0.0);
            if let Some(ray) = self.trace_lenses_from_film(&r_camera) {
                found_ray = Some(ray);
                break;
            }
        }
        if let Some(ray) = found_ray.as_ref() {
            // Compute distance _zFocus_ where ray intersects the principal axis
            let t_focus = -ray.o.x / ray.d.x;
            let mut z_focus = (ray.o + t_focus * ray.d).z;
            if z_focus < 0.0 {
                z_focus = Float::INFINITY;
            }
            return z_focus;
        } else {
            return Float::INFINITY;
        }
    }

    pub fn focus_thick_lens(&self, focus_distance: Float) -> Float {
        let (pz, fz) = self.compute_thick_lens_approximation();
        info!(
            "Cardinal points: p' = {} f' = {}, p = {} f = {}.",
            pz[0], fz[0], pz[1], fz[1]
        );
        info!("Effective focal length: {}.", fz[0] - pz[1]);
        // Compute translation of lens, _delta_, to focus at _focusDistance_
        let f = fz[0] - pz[0];
        let z = -focus_distance;
        let c = (pz[1] - z - pz[0]) * (pz[1] - z - 4.0 * f - pz[0]);
        //check
        let delta = 0.5 * (pz[1] - z + pz[0] - Float::sqrt(c));
        return self.element_interfaces.last().unwrap().thickness + delta;
    }

    pub fn compute_cardinal_points(&self, r_in: &Ray, r_out: &Ray) -> (Float, Float) {
        let tf = -r_out.o.x / r_out.d.x;
        let fz = -(r_out.o.z + tf * r_out.d.z);
        let tp = (r_in.o.x - r_out.o.x) / r_out.d.x;
        let pz = -(r_out.o.z + tp * r_out.d.z);
        return (pz, fz);
    }

    pub fn compute_thick_lens_approximation(&self) -> ([Float; 2], [Float; 2]) {
        let film = self.base.film.as_ref().read().unwrap();
        // Find height $x$ from optical axis for parallel rays
        let x = 0.001 * film.diagonal();

        // Compute cardinal points for film side of lens system
        let ro = Point3f::new(x, 0.0, self.lens_front_z() + 1.0);
        let rd = Vector3f::new(0.0, 0.0, -1.0);
        let r_scene = Ray::new(&ro, &rd, Float::INFINITY, 0.0);
        let r_film = self.trace_lenses_from_scene(&r_scene);
        let r_film = r_film.expect("Unable to trace ray from scene to film for thick lens approximation. Is aperture stop extremely small?");
        let (pz0, fz0) = self.compute_cardinal_points(&r_scene, &r_film);

        // Compute cardinal points for scene side of lens system
        let ro = Point3f::new(x, 0.0, self.lens_rear_z() - 1.0);
        let rd = Vector3f::new(0.0, 0.0, 1.0);
        let r_film = Ray::new(&ro, &rd, Float::INFINITY, 0.0);
        let r_scene = self.trace_lenses_from_film(&r_film);
        let r_scene = r_scene.expect("Unable to trace ray from film to scene for thick lens. Is aperture stop extremely small?");
        let (pz1, fz1) = self.compute_cardinal_points(&r_film, &r_scene);
        return ([pz0, pz1], [fz0, fz1]);
    }

    // helper
    fn bounds_inside(bounds: &Option<Bounds2f>, p: &Point2f) -> bool {
        if let Some(bounds) = bounds {
            return bounds.inside(p);
        }
        return false;
    }

    fn bounds_union(bounds: &Option<Bounds2f>, p: &Point2f) -> Bounds2f {
        if let Some(bounds) = bounds {
            return bounds.union_p(p);
        } else {
            return Bounds2f::new(&p, &p);
        }
    }

    pub fn bound_exit_pupil(&self, p_film_x0: Float, p_film_x1: Float) -> Bounds2f {
        let mut pupil_bounds: Option<Bounds2f> = None;
        // Sample a collection of points on the rear lens to find exit pupil
        const N_SAMPLES: u64 = 1024 * 1024;

        // Compute bounding box of projection of rear element on sampling plane
        let rear_radius = self.rear_element_radius();
        let proj_rear_bounds = Bounds2f::new(
            &Point2f::new(-1.5 * rear_radius, -1.5 * rear_radius),
            &Point2f::new(1.5 * rear_radius, 1.5 * rear_radius),
        );

        for i in 0..N_SAMPLES {
            // Find location of sample points on $x$ segment and rear lens element
            let p_film = Point3f::new(
                lerp(
                    (i as Float + 0.5) / N_SAMPLES as Float,
                    p_film_x0,
                    p_film_x1,
                ),
                0.0,
                0.0,
            );
            let u = [radical_inverse(0, i), radical_inverse(1, i)];
            let p_rear = Point3f::new(
                lerp(u[0], proj_rear_bounds.min.x, proj_rear_bounds.max.x),
                lerp(u[1], proj_rear_bounds.min.y, proj_rear_bounds.max.y),
                self.lens_rear_z(),
            );

            let p_rear2 = Point2f::new(p_rear.x, p_rear.y);
            // Expand pupil bounds if ray makes it through the lens system
            let ray = Ray::new(&p_film, &(p_rear - p_film), Float::INFINITY, 0.0);
            if Self::bounds_inside(&pupil_bounds, &p_rear2)
                || self.trace_lenses_from_film(&ray).is_some()
            {
                pupil_bounds = Some(Self::bounds_union(&pupil_bounds, &p_rear2));
            }
        }

        // Return entire element bounds if no rays made it through the lens system
        if let Some(bounds) = pupil_bounds.as_ref() {
            let delta =
                2.0 * proj_rear_bounds.diagonal().length() / Float::sqrt(N_SAMPLES as Float);
            let pupil_bounds = bounds.expand(delta);
            return pupil_bounds;
        } else {
            info!(
                "Unable to find exit pupil in x = [{}, {}]",
                p_film_x0, p_film_x1
            );
            return proj_rear_bounds;
        }
    }

    fn intersect_spherical_element(
        &self,
        radius: Float,
        z_center: Float,
        ray: &Ray,
    ) -> Option<(Float, Normal3f)> {
        // Compute _t0_ and _t1_ for ray--element intersection
        let o = ray.o - Vector3f::new(0.0, 0.0, z_center);
        let a = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
        let b = 2.0 * (ray.d.x * o.x + ray.d.y * o.y + ray.d.z * o.z);
        let c = o.x * o.x + o.y * o.y + o.z * o.z - radius * radius;
        let (t0, t1) = quadratic(a, b, c)?;

        // Select intersection $t$ based on ray direction and element curvature
        let use_closer_t = (ray.d.z > 0.0) ^ (radius < 0.0);
        let t = if use_closer_t {
            Float::min(t0, t1)
        } else {
            Float::max(t0, t1)
        };
        if t < 0.0 {
            return None;
        }
        let n = (o + t * ray.d).normalize();
        let n = face_forward(&n, &-ray.d);
        return Some((t, n));
    }

    pub fn trace_lenses_from_film(&self, r_camera: &Ray) -> Option<Ray> {
        let mut element_z = 0.0;

        let camera_to_lens = get_camera_to_lens_transform();
        let (mut r_lens, _, _) = camera_to_lens.transform_ray(r_camera);
        let element_interfaces = &self.element_interfaces;

        for i in (0..element_interfaces.len()).rev() {
            let element = &element_interfaces[i];
            // Update ray from film accounting for interaction with _element_
            element_z -= element.thickness;

            // Compute intersection of ray with lens element
            let t;
            let mut n = (-r_lens.d).normalize(); //?
            let is_stop = element.curvature_radius == 0.0;
            if is_stop {
                // The refracted ray computed in the previous lens element
                // interface may be pointed towards film plane(+z) in some
                // extreme situations; in such cases, 't' becomes negative.
                if r_lens.d.z >= 0.0 {
                    return None;
                }
                t = (element_z - r_lens.o.z) / r_lens.d.z;
            } else {
                let radius = element.curvature_radius;
                let z_center = element_z + element.curvature_radius;
                if let Some((tt, nn)) = self.intersect_spherical_element(radius, z_center, &r_lens)
                {
                    t = tt;
                    n = nn;
                } else {
                    return None;
                }
            }
            let p_hit = r_lens.o + t * r_lens.d;
            let r2 = p_hit.x * p_hit.x + p_hit.y * p_hit.y;
            if r2 > (element.aperture_radius * element.aperture_radius) {
                return None;
            }
            r_lens.o = p_hit;
            // Update ray path for element interface interaction
            if !is_stop {
                let eta_i = element.eta;
                let eta_t = if i > 0 && element_interfaces[i - 1].eta != 0.0 {
                    element_interfaces[i - 1].eta
                } else {
                    1.0
                };
                let w = refract(&(-r_lens.d.normalize()), &n, eta_i / eta_t)?;
                r_lens.d = w;
            }
        }
        // Transform _rLens_ from lens system space back to camera space
        let lens_to_camera = get_lens_to_camera_transform();
        let (r_out, _, _) = lens_to_camera.transform_ray(&r_lens);
        return Some(r_out);
    }

    pub fn trace_lenses_from_scene(&self, r_camera: &Ray) -> Option<Ray> {
        let mut element_z = -self.lens_front_z();

        let camera_to_lens = get_camera_to_lens_transform();
        let (mut r_lens, _, _) = camera_to_lens.transform_ray(r_camera);
        let element_interfaces = &self.element_interfaces;

        for i in 0..element_interfaces.len() {
            let element = &element_interfaces[i];
            // Compute intersection of ray with lens element
            let t;
            let mut n = (-r_lens.d).normalize(); //?
            let is_stop = element.curvature_radius == 0.0;
            if is_stop {
                t = (element_z - r_lens.o.z) / r_lens.d.z;
            } else {
                let radius = element.curvature_radius;
                let z_center = element_z + element.curvature_radius;
                if let Some((tt, nn)) = self.intersect_spherical_element(radius, z_center, &r_lens)
                {
                    t = tt;
                    n = nn;
                } else {
                    return None;
                }
            }
            assert!(t > 0.0);

            // Test intersection point against element aperture
            let p_hit = r_lens.o + t * r_lens.d;
            let r2 = p_hit.x * p_hit.x + p_hit.y * p_hit.y;
            if r2 > (element.aperture_radius * element.aperture_radius) {
                return None;
            }
            r_lens.o = p_hit;
            // Update ray path for element interface interaction
            if !is_stop {
                let eta_i = if i == 0 || element_interfaces[i - 1].eta == 0.0 {
                    1.0
                } else {
                    element_interfaces[i - 1].eta
                };
                let eta_t = if element_interfaces[i].eta != 0.0 {
                    element_interfaces[i].eta
                } else {
                    1.0
                };
                let w = refract(&(-r_lens.d.normalize()), &n, eta_i / eta_t)?;
                r_lens.d = w;
            }
            element_z += element.thickness;
        }
        // Transform _rLens_ from lens system space back to camera space
        let lens_to_camera = get_lens_to_camera_transform();
        let (r_out, _, _) = lens_to_camera.transform_ray(&r_lens);
        return Some(r_out);
    }
}

#[derive(Clone)]
pub struct RealisticCamera {
    base: BaseRealisticCamera,
    simple_weighting: bool,
    exit_pupil_bounds: Vec<Bounds2f>,
}

static C2L_TRANSFORM: OnceLock<Transform> = OnceLock::new();
static L2C_TRANSFORM: OnceLock<Transform> = OnceLock::new();
fn get_camera_to_lens_transform() -> &'static Transform {
    C2L_TRANSFORM.get_or_init(|| Transform::scale(1.0, 1.0, -1.0))
}
fn get_lens_to_camera_transform() -> &'static Transform {
    L2C_TRANSFORM.get_or_init(|| Transform::scale(1.0, 1.0, -1.0))
}

impl RealisticCamera {
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

        // Realistic camera-specific parameters
        let lens_file = params.get_one_filename("lensfile", "");
        let aperture_diameter = params.get_one_float("aperturediameter", 1.0);
        let focus_distance = params.get_one_float("focusdistance", 10.0);
        let simple_weighting = params.get_one_bool("simpleweighting", true);

        if lens_file == "" || !Path::new(&lens_file).exists() {
            let msg = format!("No lens description file supplied!");
            return Err(PbrtError::error(&msg));
        }
        // Load element data from lens description file

        let lens_data = read_float_file(&lens_file).map_err(|_| {
            let msg = format!("Error reading lens specification file \"{}\".", lens_file);
            PbrtError::error(&msg)
        })?;

        if (lens_data.len() % 4) != 0 {
            let msg = format!("Excess values in lens specification file \"{}\"; must be multiple-of-four values, read {}.", lens_file, lens_data.len());
            return Err(PbrtError::error(&msg));
        }

        if simple_weighting {
            warn!("\"simpleweighting\" option with RealisticCamera no longer necessarily matches regular camera images. Further, pixel values will vary a bit depending on the aperture size. See this discussion for details: https://github.com/mmp/pbrt-v3/issues/162#issuecomment-348625837");
        }

        let base_params =
            CameraBaseParameters::new(cam2world, shutteropen, shutterclose, film, medium);
        Ok(Self::new(
            base_params,
            aperture_diameter,
            focus_distance,
            simple_weighting,
            &lens_data,
        ))
    }

    pub fn new(
        base_params: CameraBaseParameters,
        aperture_diameter: Float,
        focus_distance: Float,
        simple_weighting: bool,
        lens_data: &[Float],
    ) -> Self {
        let base = BaseCamera::new(base_params);
        let mut element_interfaces = Vec::new();
        for i in (0..lens_data.len()).step_by(4) {
            let mut data = lens_data[i..(i + 4)].to_vec();
            if data[0] == 0.0 {
                if aperture_diameter > data[3] {
                    warn!("Specified aperture diameter {} is greater than maximum possible {}.  Clamping it.",
                          aperture_diameter, data[3]);
                } else {
                    data[3] = aperture_diameter;
                }
            }
            //# radius	sep	n	aperture
            let lens = LensElementInterface {
                curvature_radius: data[0] * 0.001,
                thickness: data[1] * 0.001,
                eta: data[2],
                aperture_radius: data[3] * 0.001 / 2.0,
            };
            element_interfaces.push(lens);
        }

        // Compute lens--film distance for given focus distance
        let mut base = BaseRealisticCamera {
            base,
            element_interfaces,
        };
        let fb1 = base.focus_binary_search(focus_distance);
        info!("Binary search focus: {} -> ", fb1);

        let old_fd = base.element_interfaces.last().unwrap().thickness;
        let new_fd = base.focus_thick_lens(focus_distance);
        info!("Thick lens focus: {} -> {}", old_fd, new_fd);
        base.element_interfaces.last_mut().unwrap().thickness = new_fd;

        // Compute exit pupil bounds at sampled points on the film
        const N_SAMPLES: usize = 64;
        let diagonal = base.base.film.as_ref().read().unwrap().diagonal();
        let mut exit_pupil_bounds = Vec::with_capacity(N_SAMPLES);
        for i in 0..N_SAMPLES {
            let r0 = (i + 0) as Float / N_SAMPLES as Float * diagonal / 2.0;
            let r1 = (i + 1) as Float / N_SAMPLES as Float * diagonal / 2.0;
            exit_pupil_bounds.push(base.bound_exit_pupil(r0, r1));
        }

        RealisticCamera {
            base,
            simple_weighting,
            exit_pupil_bounds,
        }
    }

    // private methods
    fn lens_rear_z(&self) -> Float {
        return self.base.lens_rear_z();
    }
    fn sample_exit_pupil(&self, p_film: &Point2f, lens_sample: &Point2f) -> (Point3f, Float) {
        // Find exit pupil bound for sample distance from film center
        let diagonal = self.base.base.film.as_ref().read().unwrap().diagonal();
        let exit_pupil_bounds = &self.exit_pupil_bounds;
        let r_film = Float::sqrt(p_film.x * p_film.x + p_film.y * p_film.y);
        let r_index = (r_film / (diagonal / 2.0) * exit_pupil_bounds.len() as Float) as usize;
        let r_index = usize::min(exit_pupil_bounds.len() - 1, r_index);
        let pupil_bounds = &exit_pupil_bounds[r_index];
        let sample_bounds_area = pupil_bounds.area();

        // Generate sample point inside exit pupil bound
        let p_lens = pupil_bounds.lerp(lens_sample);

        // Return sample point rotated by angle of _pFilm_ with $+x$ axis
        let sin_theta = if r_film != 0.0 {
            p_film.y / r_film
        } else {
            0.0
        };
        let cos_theta = if r_film != 0.0 {
            p_film.x / r_film
        } else {
            1.0
        };
        let p = Point3f::new(
            cos_theta * p_lens.x - sin_theta * p_lens.y,
            sin_theta * p_lens.x + cos_theta * p_lens.y,
            self.lens_rear_z(),
        );

        return (p, sample_bounds_area);
    }

    fn find_point_on_film(&self, sample: &CameraSample) -> Point2f {
        let film = self.base.base.film.as_ref().read().unwrap();
        // Find point on film, _pFilm_, corresponding to _sample.pFilm_
        let full_resolution = film.full_resolution();
        let s = Vector2f::new(
            sample.p_film.x / full_resolution.x as Float,
            sample.p_film.y / full_resolution.y as Float,
        );
        let p_film2 = film.get_physical_extent().lerp(&s);
        let p_film = Vector2f::new(-p_film2.x, p_film2.y);
        return p_film;
    }
}

unsafe impl Sync for RealisticCamera {}

impl RealisticCamera {
    fn film_point_to_raster(&self, p_film: &Point2f) -> Option<Point2f> {
        let film = self.base.base.film.as_ref().read().unwrap();
        let p_film2 = Point2f::new(-p_film.x, p_film.y);
        let s = film.get_physical_extent().offset(&p_film2);
        let full_resolution = film.full_resolution();
        let p_raster = Point2f::new(
            s.x * full_resolution.x as Float,
            s.y * full_resolution.y as Float,
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

    fn evaluate_importance_geometry(&self, ray: &Ray) -> Option<(Point2f, Float, Float)> {
        let c2w = self.base.base.camera_to_world.interpolate(ray.time);
        let w2c = c2w.inverse();

        let d_camera = w2c.transform_vector(&ray.d);
        if d_camera.length_squared() == 0.0 {
            return None;
        }
        let d_camera = d_camera.normalize();
        if d_camera.z <= 0.0 {
            return None;
        }

        let o_camera = w2c.transform_point(&ray.o);
        let r_scene = Ray::new(
            &(o_camera + 1e-3 * d_camera),
            &(-d_camera),
            Float::INFINITY,
            ray.time,
        );
        let r_film = self.base.trace_lenses_from_scene(&r_scene)?;
        if Float::abs(r_film.d.z) < 1e-6 {
            return None;
        }
        let t = -r_film.o.z / r_film.d.z;
        if t <= 0.0 {
            return None;
        }
        let p_film3 = r_film.position(t);
        let p_film = Point2f::new(p_film3.x, p_film3.y);
        let p_raster = self.film_point_to_raster(&p_film)?;

        let cos_theta = Float::abs(d_camera.z);
        if cos_theta <= 0.0 {
            return None;
        }

        let diagonal = self.base.base.film.as_ref().read().unwrap().diagonal();
        let r_film = Float::sqrt(p_film.x * p_film.x + p_film.y * p_film.y);
        let mut r_index =
            (r_film / (diagonal / 2.0) * self.exit_pupil_bounds.len() as Float) as usize;
        r_index = usize::min(self.exit_pupil_bounds.len() - 1, r_index);
        let exit_pupil_area = self.exit_pupil_bounds[r_index].area();
        if exit_pupil_area <= 0.0 {
            return None;
        }

        Some((p_raster, cos_theta, exit_pupil_area))
    }

    pub fn generate_ray(
        &self,
        sample: &CameraSample,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraRay> {
        let _p = ProfilePhase::new(Prof::GenerateCameraRay);

        RAYS.with(|rays| {
            //totalRays
            rays.add_denom(1);
        });

        let p_film = self.find_point_on_film(sample);

        // Trace ray from _pFilm_ through lens system
        let (p_rear, exit_pupil_bounds_area) = self.sample_exit_pupil(&p_film, &sample.p_lens);
        let p_film = Vector3f::new(p_film.x, p_film.y, 0.0);

        let time = lerp(
            sample.time,
            self.base.base.shutter_open,
            self.base.base.shutter_close,
        );

        let r_film = Ray::new(&p_film, &(p_rear - p_film), Float::INFINITY, time);
        assert!(r_film.d.length_squared() > 0.0);
        let ray = if let Some(ray) = self.base.trace_lenses_from_film(&r_film) {
            ray
        } else {
            RAYS.with(|rays| {
                //vignettedRays
                rays.add_num(1);
            });
            return None;
        };

        // Finish initialization of _RealisticCamera_ ray
        let (mut ray, _, _) = self.base.base.camera_to_world.transform_ray(&ray);
        ray.d = ray.d.normalize();
        ray.medium = self.get_medium();

        // Return weighting for _RealisticCamera_ ray
        let cos_theta = r_film.d.normalize().z;
        let cos_4_theta = (cos_theta * cos_theta) * (cos_theta * cos_theta);
        let weight = if self.simple_weighting {
            cos_4_theta * exit_pupil_bounds_area / self.exit_pupil_bounds[0].area()
        } else {
            (self.base.base.shutter_close - self.base.base.shutter_open)
                * cos_4_theta
                * exit_pupil_bounds_area
                / (self.lens_rear_z() * self.lens_rear_z())
        };

        return Some(CameraRay { ray, weight });
    }

    pub fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        BaseCamera::generate_ray_differential(&Camera::Realistic(self.clone()), sample, lambda)
    }

    pub fn we(&self, ray: &Ray) -> Option<(Spectrum, Point2f)> {
        let (p_raster, cos_theta, exit_pupil_area) = self.evaluate_importance_geometry(ray)?;
        let cos4_theta = (cos_theta * cos_theta) * (cos_theta * cos_theta);
        if cos4_theta <= 0.0 {
            return None;
        }

        let weight = if self.simple_weighting {
            self.exit_pupil_bounds[0].area() / (exit_pupil_area * cos4_theta)
        } else {
            let shutter = self.base.base.shutter_close - self.base.base.shutter_open;
            if shutter <= 0.0 {
                return None;
            }
            (self.lens_rear_z() * self.lens_rear_z()) / (shutter * exit_pupil_area * cos4_theta)
        };

        if !weight.is_finite() || weight <= 0.0 {
            return None;
        }
        Some((Spectrum::from(weight), p_raster))
    }

    pub fn pdf_we(&self, ray: &Ray) -> Option<(Float, Float)> {
        let (_p_raster, cos_theta, exit_pupil_area) = self.evaluate_importance_geometry(ray)?;
        let cos3_theta = cos_theta * cos_theta * cos_theta;
        if cos3_theta <= 0.0 {
            return None;
        }
        let pdf_pos = 1.0 / exit_pupil_area;
        let pdf_dir = (self.lens_rear_z() * self.lens_rear_z()) / (exit_pupil_area * cos3_theta);
        if !pdf_pos.is_finite() || !pdf_dir.is_finite() || pdf_pos <= 0.0 || pdf_dir <= 0.0 {
            return None;
        }
        Some((pdf_pos, pdf_dir))
    }

    pub fn sample_wi(
        &self,
        inter: &Interaction,
        u: &Point2f,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraWiSample> {
        let c2w = self.base.base.camera_to_world.interpolate(inter.get_time());
        let rear_radius = self.base.rear_element_radius();
        if rear_radius <= 0.0 {
            return None;
        }

        let p_lens = rear_radius * concentric_sample_disk(u);
        let p_lens_camera = Point3f::new(p_lens.x, p_lens.y, self.lens_rear_z());
        let p_lens_world = c2w.transform_point(&p_lens_camera);

        let wi = p_lens_world - inter.get_p();
        let dist2 = wi.length_squared();
        if dist2 == 0.0 {
            return None;
        }
        let wi = wi / Float::sqrt(dist2);

        let n = Normal3f::from(c2w.transform_vector(&Vector3f::new(0.0, 0.0, 1.0)));
        let cos = n.abs_dot(&wi);
        if cos <= 0.0 {
            return None;
        }

        let medium = self.base.base.medium.clone();
        let lens_intr = Interaction::Base(BaseInteraction {
            p: p_lens_world,
            time: inter.get_time(),
            medium_interface: MediumInterface::from(&medium),
            n,
            ..Default::default()
        });
        let vis = VisibilityTester::from((inter.clone(), lens_intr.clone()));
        let ray = lens_intr.spawn_ray(&-wi);
        let (spec, p_raster) = self.we(&ray)?;

        let lens_area = PI * rear_radius * rear_radius;
        if lens_area <= 0.0 {
            return None;
        }
        let pdf = dist2 / (cos * lens_area);
        if !pdf.is_finite() || pdf <= 0.0 {
            return None;
        }

        Some(CameraWiSample {
            wi_spec: spec,
            wi,
            pdf,
            p_raster,
            visibility: vis,
        })
    }

    pub fn get_film(&self) -> Arc<RwLock<Film>> {
        return self.base.base.get_film();
    }

    pub fn get_medium(&self) -> Option<Arc<Medium>> {
        return self.base.base.get_medium();
    }

    pub fn get_shutter(&self) -> (Float, Float) {
        return self.base.base.get_shutter();
    }

    pub fn get_camera_to_world(&self) -> AnimatedTransform {
        self.base.base.get_camera_to_world()
    }

    pub fn init_minimum_differentials(&mut self) {
        let differentials =
            BaseCamera::find_minimum_differentials(&Camera::Realistic(self.clone()));
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
