// pbrt-v4 verbatim translation of `class RayIntegrator` and its
// associated `EvaluatePixelSample` plus the `ImageTileIntegrator` driver.
//
// Reference: pbrt-v4/src/pbrt/cpu/integrators.h:80-112 and
// pbrt-v4/src/pbrt/cpu/integrators.cpp:66-289.
//
// r4 keeps the same inheritance chain Integrator -> ImageTileIntegrator
// -> RayIntegrator. Construction order, field names and method
// signatures all mirror v4. `MemoryArena` is the Rust analogue of v4's
// `ScratchBuffer`; it persists for the trait surface but is otherwise
// vestigial (Rust does not need an explicit arena).

use super::integrator::*;
use crate::base::camera::{Camera, CameraRayDifferential, CameraSample};
use crate::base::sampler::Sampler;
use crate::film::*;
use crate::interaction::*;
use crate::options::PbrtOptions;
use crate::scene::*;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::memory::*;
use crate::util::misc::*;
use crate::util::spectrum::*;
use crate::util::stats::pixel_stats;

use std::ops::DerefMut;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;

use log::*;
use rayon::prelude::*;

/// pbrt-v4 `class RayIntegrator : public ImageTileIntegrator`. Its only
/// abstract method is `Li`; everything else lives on the base. Concrete
/// integrators (PathIntegrator, AOIntegrator, VolPathIntegrator, ...)
/// override only `Li`. The `MemoryArena` argument is the Rust analogue of
/// v4's `ScratchBuffer &scratchBuffer`.
pub trait RayIntegrator: ImageTileIntegrator {
    fn preprocess(&mut self, _sampler: &mut Sampler) {}

    /// pbrt-v4 `virtual SampledSpectrum Li(RayDifferential ray,
    /// SampledWavelengths &lambda, Sampler sampler, ScratchBuffer
    /// &scratchBuffer, VisibleSurface *visibleSurface) const = 0`.
    fn li(
        &self,
        ray: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut MemoryArena,
        visible_surface: Option<&mut VisibleSurface>,
    ) -> SampledSpectrum;

    fn get_sampler(&self) -> Arc<RwLock<Sampler>>;
    fn get_pixel_bounds(&self) -> Bounds2i;
}

/// pbrt-v4 `void RayIntegrator::EvaluatePixelSample(Point2i pPixel, int
/// sampleIndex, Sampler sampler, ScratchBuffer &scratchBuffer)`
/// (integrators.cpp:227). Translated line-by-line.
pub fn evaluate_pixel_sample_ray_default<I: RayIntegrator + ?Sized>(
    integrator: &I,
    p_pixel: Point2i,
    sample_index: i32,
    sampler: &mut Sampler,
    scratch_buffer: &mut MemoryArena,
) -> Option<PixelSample> {
    let camera_arc = integrator.get_camera();
    let camera = camera_arc.as_ref();
    let film_arc = camera.get_film();

    // Sample wavelengths for the ray
    let lu = if PbrtOptions::get().disable_wavelength_jitter {
        0.5
    } else {
        sampler.get_1d()
    };
    let (uses_visible_surface, gbuffer_coordinate_system, filter, mut lambda) = {
        let film = film_arc.read().unwrap();
        let lambda = film.sample_wavelengths(lu);
        (
            film.uses_visible_surface(),
            film.gbuffer_coordinate_system(),
            film.filter().clone(),
            lambda,
        )
    };

    // Initialize CameraSample for current sample. pbrt-v4 samples the
    // reconstruction filter directly and adds the result only to p_pixel.
    let fs = filter.sample(&sampler.get_pixel_2d());
    let camera_sample = CameraSample {
        p_film: Point2f::new(
            p_pixel.x as Float + fs.p.x + 0.5,
            p_pixel.y as Float + fs.p.y + 0.5,
        ),
        time: sampler.get_1d(),
        p_lens: sampler.get_2d(),
        filter_weight: fs.weight,
    };
    let camera_sample = if PbrtOptions::get().disable_pixel_jitter {
        CameraSample {
            p_film: Point2f::new(p_pixel.x as Float + 0.5, p_pixel.y as Float + 0.5),
            time: 0.5,
            p_lens: Point2f::new(0.5, 0.5),
            filter_weight: 1.0,
        }
    } else {
        camera_sample
    };

    // Generate camera ray for current sample
    let camera_ray_opt = camera.generate_ray_differential(&camera_sample, &lambda);

    // Trace cameraRay if valid
    let mut l = SampledSpectrum::zero();
    let mut visible_surface = VisibleSurface::default();
    let ray_weight;
    if let Some(camera_ray) = camera_ray_opt {
        let CameraRayDifferential { mut ray, weight } = camera_ray;
        ray_weight = weight;
        // Scale camera ray differentials based on image sampling rate
        if !PbrtOptions::get().disable_pixel_jitter {
            let samples_per_pixel = sampler.samples_per_pixel() as Float;
            let ray_diff_scale = Float::max(0.125, 1.0 / Float::sqrt(samples_per_pixel));
            ray.scale_differentials(ray_diff_scale);
        }

        // Evaluate radiance along camera ray
        l = weight
            * integrator.li(
                &ray,
                &mut lambda,
                sampler,
                scratch_buffer,
                if uses_visible_surface {
                    Some(&mut visible_surface)
                } else {
                    None
                },
            );

        // Issue warning if unexpected radiance value is returned
        if l.has_nans() {
            error!(
                "Not-a-number radiance value returned for pixel ({}, {}), sample {}. Setting to black.",
                p_pixel.x, p_pixel.y, sample_index
            );
            l = SampledSpectrum::zero();
        } else if l.y(&lambda).is_infinite() {
            error!(
                "Infinite radiance value returned for pixel ({}, {}), sample {}. Setting to black.",
                p_pixel.x, p_pixel.y, sample_index
            );
            l = SampledSpectrum::zero();
        }
    } else {
        ray_weight = 0.0;
    }

    let transformed_visible_surface = if visible_surface.set {
        if uses_visible_surface && gbuffer_coordinate_system == GBufferCoordinateSystem::Camera {
            let world_to_camera = camera.get_camera_to_world().inverse();
            world_to_camera.map(|transform| visible_surface.transformed(&transform))
        } else {
            Some(visible_surface)
        }
    } else {
        None
    };

    Some(PixelSample {
        p_pixel,
        p_film: camera_sample.p_film,
        l,
        lambda,
        ray_weight,
        visible_surface: transformed_visible_surface,
        filter_weight: camera_sample.filter_weight,
    })
}

/// pbrt-v4 `class ImageTileIntegrator : public Integrator`. Its only
/// abstract method is `EvaluatePixelSample`. The tile loop lives in
/// `ImageTileIntegrator::Render` (integrators.cpp:66) and is provided
/// by `SampleIntegratorCore::render` below.
pub struct ImageTileIntegratorBase {
    pub base: IntegratorBase,
    pub camera: Arc<Camera>,
    pub sampler: Arc<RwLock<Sampler>>,
}

impl ImageTileIntegratorBase {
    pub fn new(scene: &Scene, camera: &Arc<Camera>, sampler: &Arc<RwLock<Sampler>>) -> Self {
        ImageTileIntegratorBase {
            base: IntegratorBase::from_scene(scene),
            camera: Arc::clone(camera),
            sampler: Arc::clone(sampler),
        }
    }
}

impl std::ops::Deref for ImageTileIntegratorBase {
    type Target = IntegratorBase;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl std::ops::DerefMut for ImageTileIntegratorBase {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

/// Output of `ImageTileIntegrator::EvaluatePixelSample`. Mirrors v4's
/// `Film::AddSample(pPixel, L, lambda, visibleSurface, filterWeight)`
/// argument bundle (integrators.cpp:287). The tile loop uses `p_pixel`
/// as the accumulation target; `p_film` is the filter-sampled camera
/// position used for ray generation. `None` means the integrator handled its own splat
/// (LightPathIntegrator) and the tile loop should skip the FilmTile
/// update for that sample.
#[derive(Clone)]
pub struct PixelSample {
    pub p_pixel: Point2i,
    pub p_film: Point2f,
    pub l: SampledSpectrum,
    pub lambda: SampledWavelengths,
    pub ray_weight: Float,
    pub visible_surface: Option<VisibleSurface>,
    pub filter_weight: Float,
}

pub trait ImageTileIntegrator: Integrator + Sync {
    fn evaluate_pixel_sample(
        &self,
        p_pixel: Point2i,
        sample_index: i32,
        sampler: &mut Sampler,
        scratch_buffer: &mut MemoryArena,
    ) -> Option<PixelSample>;
}

/// Boilerplate impl forwarding `evaluate_pixel_sample` to the RayIntegrator
/// default. Every RayIntegrator type calls this macro.
#[macro_export]
macro_rules! impl_image_tile_integrator_via_ray {
    ($t:ty) => {
        impl $crate::cpu::integrators::ImageTileIntegrator for $t {
            fn evaluate_pixel_sample(
                &self,
                p_pixel: $crate::util::base::Point2i,
                sample_index: i32,
                sampler: &mut $crate::base::sampler::Sampler,
                scratch_buffer: &mut $crate::util::memory::MemoryArena,
            ) -> Option<$crate::cpu::integrators::PixelSample> {
                $crate::cpu::integrators::evaluate_pixel_sample_ray_default(
                    self,
                    p_pixel,
                    sample_index,
                    sampler,
                    scratch_buffer,
                )
            }
        }
    };
}

pub struct RayIntegratorBase {
    pub base: ImageTileIntegratorBase,
    pub pixel_bounds: Bounds2i,
}

impl std::ops::Deref for RayIntegratorBase {
    type Target = ImageTileIntegratorBase;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl std::ops::DerefMut for RayIntegratorBase {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl RayIntegratorBase {
    pub fn new(
        scene: &Scene,
        camera: &Arc<Camera>,
        sampler: &Arc<RwLock<Sampler>>,
        pixel_bounds: &Bounds2i,
    ) -> Self {
        RayIntegratorBase {
            base: ImageTileIntegratorBase::new(scene, camera, sampler),
            pixel_bounds: pixel_bounds.clone(),
        }
    }

    pub fn render(integrator: &mut dyn RayIntegrator) {
        let acamera = integrator.get_camera();
        let camera = acamera.as_ref();
        let sampler = integrator.get_sampler();
        {
            let mut sampler = sampler.write().unwrap();
            integrator.preprocess(sampler.deref_mut());
        }

        let film = camera.get_film();
        SampleIntegratorCore::render(integrator, camera, &film, &sampler);
    }
}

pub struct SampleIntegratorCore {}

impl SampleIntegratorCore {
    fn get_film_tile(film: &Arc<RwLock<Film>>, tile_bounds: &Bounds2i) -> FilmTile {
        let film = film.read().unwrap();
        film.get_film_tile(tile_bounds)
    }

    fn merge_film_tile(film: &Arc<RwLock<Film>>, tile: &FilmTile) {
        {
            let mut f = film.write().unwrap();
            f.merge_film_tile(tile);
        }
        let f = film.read().unwrap();
        f.update_display(&tile.get_pixel_bounds());
    }

    fn get_filename(film: &Arc<RwLock<Film>>) -> String {
        let film = film.read().unwrap();
        let path = film.filename();
        let path = Path::new(&path);
        let filename = path.file_name().unwrap();
        filename.to_str().unwrap().to_string()
    }

    fn render_tile(
        integrator: &dyn ImageTileIntegrator,
        _camera: &Camera,
        film: &Arc<RwLock<Film>>,
        tile_bounds: &Bounds2i,
        sampler: &Arc<Mutex<Sampler>>,
        reporter: &Arc<Mutex<ProgressReporter>>,
    ) {
        let mut scratch_buffer = MemoryArena::new();
        let x0 = tile_bounds.min.x;
        let x1 = tile_bounds.max.x;
        let y0 = tile_bounds.min.y;
        let y1 = tile_bounds.max.y;

        let mut sampler = sampler.lock().unwrap();

        let samples_per_pixel = sampler.samples_per_pixel();
        let mut film_tile = Self::get_film_tile(film, tile_bounds);
        for yy in y0..y1 {
            for xx in x0..x1 {
                let pixel = Point2i::new(xx, yy);
                pixel_stats::report_pixel_start(pixel);
                for sample_index in 0..samples_per_pixel {
                    sampler.start_pixel_sample(pixel, sample_index, 0);
                    if let Some(s) = integrator.evaluate_pixel_sample(
                        pixel,
                        sample_index as i32,
                        &mut *sampler,
                        &mut scratch_buffer,
                    ) {
                        film_tile.add_sample_pixel(
                            &s.p_pixel,
                            s.l,
                            &s.lambda,
                            s.visible_surface.as_ref(),
                            s.filter_weight,
                        );
                    }
                    scratch_buffer.reset();
                }
                pixel_stats::report_pixel_end(pixel);
            }
        }
        Self::merge_film_tile(film, &film_tile);
        {
            let mut reporter = reporter.lock().unwrap();
            reporter.update(1);
        }
    }

    pub fn render(
        integrator: &dyn ImageTileIntegrator,
        camera: &Camera,
        film: &Arc<RwLock<Film>>,
        sampler: &Arc<RwLock<Sampler>>,
    ) {
        let mut tile_indices = Vec::new();
        {
            let film_r = film.as_ref().read().unwrap();
            let pixel_bounds = film_r.pixel_bounds();
            let sample_extent = pixel_bounds.diagonal();
            const TILE_SIZE: i32 = 16;
            let n_tiles = Point2i::from((
                (sample_extent.x + TILE_SIZE - 1) / TILE_SIZE,
                (sample_extent.y + TILE_SIZE - 1) / TILE_SIZE,
            ));
            tile_indices.reserve((n_tiles.x * n_tiles.y) as usize);
            for y in 0..n_tiles.y {
                for x in 0..n_tiles.x {
                    let x0 = pixel_bounds.min.x + x * TILE_SIZE;
                    let x1 = i32::min(x0 + TILE_SIZE, pixel_bounds.max.x);
                    let y0 = pixel_bounds.min.y + y * TILE_SIZE;
                    let y1 = i32::min(y0 + TILE_SIZE, pixel_bounds.max.y);
                    let tile_bounds = Bounds2i::from(((x0, y0), (x1, y1)));

                    let s = sampler.read().unwrap().clone();
                    let s_arc = Arc::new(Mutex::new(s));
                    tile_indices.push((tile_bounds, s_arc));
                }
            }

            if PbrtOptions::get().record_pixel_statistics {
                pixel_stats::enable(film_r.pixel_bounds(), film_r.filename());
            }
            film_r.render_start();
        }

        {
            let filename = Self::get_filename(film);

            let total = tile_indices.len();
            let reporter = Arc::new(Mutex::new(ProgressReporter::new(total, &filename)));

            tile_indices.par_iter().for_each(|(tile_bounds, sampler)| {
                Self::render_tile(integrator, camera, film, tile_bounds, sampler, &reporter);
            });

            {
                let mut reporter = reporter.lock().unwrap();
                reporter.done();
            }
        }

        {
            let film_r = film.as_ref().read().unwrap();
            film_r.render_end();
            // pbrt-v4 `ImageTileIntegrator::Render` (integrators.cpp:215)
            // writes the image with `WriteImage(metadata, 1/waveStart)`,
            // where the second argument is the splat-scale that converts
            // per-pixel accumulated splats (BDPT t=1, LightPath, etc.) to
            // per-sample averages. r4 stores splats in tiles, so we flush
            // them into `splat_pixels` here with the matching scale before
            // `write_image` produces the final EXR. Integrators that
            // never call `add_splat` (path / volpath) leave every tile
            // clean and pay no cost.
            let spp = {
                let sampler_proto = sampler
                    .read()
                    .expect("RayIntegrator: poisoned sampler RwLock");
                sampler_proto.samples_per_pixel() as Float
            };
            let splat_scale = if spp > 0.0 { 1.0 / spp } else { 1.0 };
            film_r.merge_splats(splat_scale);
            film_r.write_image();
            if PbrtOptions::get().record_pixel_statistics {
                if let Err(error) = pixel_stats::write() {
                    error!("Failed to write pixel statistics: {}", error);
                }
            }
        }
    }
}

/// Validate a sampled-spectrum radiance result before it's handed to the
/// film. Mirrors the NaN / negative / infinite checks in v4
/// `RayIntegrator::EvaluatePixelSample` (integrators.cpp:263-273).
pub fn validate_sampled_radiance(
    l: SampledSpectrum,
    pixel: &Point2i,
    lambda: &SampledWavelengths,
) -> SampledSpectrum {
    if !l.is_valid() {
        error!(
            "Not-a-number radiance value returned for pixel ({}, {}). Setting to black.",
            pixel.x, pixel.y
        );
        return SampledSpectrum::zero();
    }
    let y = l.y(lambda);
    if y < -1e-5 {
        error!(
            "Negative luminance value, {}, returned for pixel ({}, {}). Setting to black.",
            y, pixel.x, pixel.y
        );
        return SampledSpectrum::zero();
    }
    if y.is_infinite() {
        error!(
            "Infinite luminance value returned for pixel ({}, {}). Setting to black.",
            pixel.x, pixel.y
        );
        return SampledSpectrum::zero();
    }
    l
}
