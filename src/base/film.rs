use crate::base::filter::Filter;
use crate::displays::*;
use crate::film::film_base::{FilmBase, FilmBaseParameters};
use crate::film::film_tile::FilmTile;
use crate::film::gbuffer_film::GBufferFilm;
use crate::film::pixel_sensor::PixelSensor;
use crate::film::rgb_film::RGBFilm;
use crate::film::spectral_film::{
    SpectralFilm, SPECTRAL_LAMBDA_MAX_DEFAULT, SPECTRAL_LAMBDA_MIN_DEFAULT,
    SPECTRAL_NUM_BUCKETS_DEFAULT,
};
use crate::options::*;
use crate::paramdict::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::image::Image;
use crate::util::profile::*;
use crate::util::spectrum::*;

use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GBufferCoordinateSystem {
    Camera,
    World,
}

pub enum Film {
    Rgb(RGBFilm),
    GBuffer(GBufferFilm),
    Spectral(SpectralFilm),
}

impl Film {
    fn get_full_path(filename: &str) -> String {
        let path = Path::new(filename);
        if path.is_absolute() {
            path.to_string_lossy().into_owned()
        } else {
            let dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let full_path = dir.join(path);
            full_path.to_string_lossy().into_owned()
        }
    }

    fn default_pixel_bounds(full_resolution: &Point2i) -> Bounds2i {
        Bounds2i::from(((0, 0), (full_resolution.x, full_resolution.y)))
    }

    fn pixel_bounds_from_crop_window(crop: &Bounds2f, full_resolution: &Point2i) -> Bounds2i {
        // v4 verbatim: `pixelBounds = Bounds2i(ceil(fullRes * crop.pMin),
        // ceil(fullRes * crop.pMax))`. For a cropwindow narrow enough
        // that `ceil(fullRes * pMin)` and `ceil(fullRes * pMax)` round
        // to the same integer (e.g. cropwindow `0.49 0.51` on a
        // 45-pixel-tall film), v4 happily produces a 0-area bounds.
        // r4 then errors out with "Degenerate pixel bounds". Match v4's
        // intent — clamp to at least 1×1 by extending the max side, or
        // pulling the min side back when the max is already at the
        // full-resolution edge.
        let mut min_x = Float::ceil(full_resolution.x as Float * crop.min.x) as i32;
        let mut min_y = Float::ceil(full_resolution.y as Float * crop.min.y) as i32;
        let mut max_x = Float::ceil(full_resolution.x as Float * crop.max.x) as i32;
        let mut max_y = Float::ceil(full_resolution.y as Float * crop.max.y) as i32;
        if max_x <= min_x {
            if max_x < full_resolution.x {
                max_x = min_x + 1;
            } else {
                min_x = (max_x - 1).max(0);
            }
        }
        if max_y <= min_y {
            if max_y < full_resolution.y {
                max_y = min_y + 1;
            } else {
                min_y = (max_y - 1).max(0);
            }
        }
        Bounds2i::from(((min_x, min_y), (max_x, max_y)))
    }

    fn get_crop_window(params: &ParameterDictionary) -> Result<Option<Bounds2f>, PbrtError> {
        if let Some(cropwindow) = params.get_floats_ref("cropwindow") {
            if cropwindow.len() != 4 {
                return Err(PbrtError::error(&format!(
                    "{} values supplied for \"cropwindow\". Expected 4.",
                    cropwindow.len()
                )));
            }

            let crop = Bounds2f::from((
                (
                    Float::clamp(Float::min(cropwindow[0], cropwindow[1]), 0.0, 1.0),
                    Float::clamp(Float::min(cropwindow[2], cropwindow[3]), 0.0, 1.0),
                ),
                (
                    Float::clamp(Float::max(cropwindow[0], cropwindow[1]), 0.0, 1.0),
                    Float::clamp(Float::max(cropwindow[2], cropwindow[3]), 0.0, 1.0),
                ),
            ));
            Ok(Some(crop))
        } else {
            Ok(None)
        }
    }

    fn get_pixel_bounds_from_params(
        params: &ParameterDictionary,
        full_resolution: &Point2i,
    ) -> Result<Bounds2i, PbrtError> {
        let mut pixel_bounds = Self::default_pixel_bounds(full_resolution);

        if let Some(pb) = params.get_ints_ref("pixelbounds") {
            if pb.len() != 4 {
                return Err(PbrtError::error(&format!(
                    "{} values supplied for \"pixelbounds\". Expected 4.",
                    pb.len()
                )));
            }
            let requested = Bounds2i::from(((pb[0], pb[2]), (pb[1], pb[3])));
            pixel_bounds = requested.intersect(&pixel_bounds);
        }

        if let Some(crop) = Self::get_crop_window(params)? {
            pixel_bounds = Self::pixel_bounds_from_crop_window(&crop, full_resolution);
        }

        if pixel_bounds.area() <= 0 {
            return Err(PbrtError::error(&format!(
                "Degenerate pixel bounds provided to film: {:?}.",
                pixel_bounds
            )));
        }

        Ok(pixel_bounds)
    }

    pub fn create(
        name: &str,
        params: &ParameterDictionary,
        filter: &Filter,
    ) -> Result<Arc<RwLock<Film>>, PbrtError> {
        let filename = params.get_one_string("filename", "pbrt.exr");
        let filepath = Self::get_full_path(&filename);

        let mut xres = params.get_one_int("xresolution", 1280);
        let mut yres = params.get_one_int("yresolution", 720);
        let options = PbrtOptions::get();
        if options.quick_render && !options.quick_render_full_resolution {
            xres = i32::max(1, xres / 4);
            yres = i32::max(1, yres / 4);
        }

        let resolution = Point2i::from((xres, yres));
        let pixel_bounds = Self::get_pixel_bounds_from_params(params, &resolution)?;
        let scale = params.get_one_float("scale", 1.0);
        let diagonal = params.get_one_float("diagonal", 35.0);
        let maxcomponentvalue = params.get_one_float("maxcomponentvalue", Float::INFINITY);
        let iso = params.get_one_float("iso", 100.0);
        let white_balance = params.get_one_float("whitebalance", 0.0);
        let sensor_name = params.get_one_string("sensor", "cie1931");
        let pixel_sensor = PixelSensor::create(&sensor_name, iso, white_balance)?;
        let gbuffer_coordinate_system =
            match params.get_one_string("coordinatesystem", "camera").as_str() {
                "camera" => GBufferCoordinateSystem::Camera,
                "world" => GBufferCoordinateSystem::World,
                value => {
                    return Err(PbrtError::error(&format!(
                        "Unknown gbuffer coordinate system \"{}\".",
                        value
                    )));
                }
            };

        let film = match name {
            "rgb" => Film::new_rgb(
                &resolution,
                &pixel_bounds,
                filter,
                diagonal,
                &filepath,
                scale,
                maxcomponentvalue,
                pixel_sensor,
            ),
            "gbuffer" => {
                let ext = Path::new(&filepath)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                if ext != "exr" {
                    return Err(PbrtError::error(
                        "EXR is the only format supported by the gbuffer film.",
                    ));
                }
                Film::new_gbuffer(
                    &resolution,
                    &pixel_bounds,
                    filter,
                    diagonal,
                    &filepath,
                    scale,
                    maxcomponentvalue,
                    gbuffer_coordinate_system,
                    pixel_sensor,
                )
            }
            "spectral" => {
                // pbrt-v4 SpectralFilm::Create reads lambdamin/lambdamax/nbuckets;
                // see src/pbrt/film.cpp.
                let lambda_min = params.get_one_float("lambdamin", SPECTRAL_LAMBDA_MIN_DEFAULT);
                let lambda_max = params.get_one_float("lambdamax", SPECTRAL_LAMBDA_MAX_DEFAULT);
                let n_buckets =
                    params.get_one_int("nbuckets", SPECTRAL_NUM_BUCKETS_DEFAULT as i32) as usize;
                Film::Spectral(SpectralFilm::with_spectral_range(
                    FilmBaseParameters {
                        full_resolution: resolution,
                        pixel_bounds,
                        filter: filter.clone(),
                        diagonal,
                        filename: filepath.clone(),
                        pixel_sensor,
                        scale,
                        max_sample_luminance: maxcomponentvalue,
                    },
                    lambda_min,
                    lambda_max,
                    n_buckets,
                ))
            }
            _ => {
                return Err(PbrtError::error(&format!("Film \"{}\" unknown.", name)));
            }
        };

        Ok(Arc::new(RwLock::new(film)))
    }

    pub fn new_rgb(
        resolution: &Point2i,
        pixel_bounds: &Bounds2i,
        filter: &Filter,
        diagonal: Float,
        filename: &str,
        scale: Float,
        max_sample_luminance: Float,
        pixel_sensor: PixelSensor,
    ) -> Self {
        Self::Rgb(RGBFilm::new(FilmBaseParameters {
            full_resolution: *resolution,
            pixel_bounds: *pixel_bounds,
            filter: filter.clone(),
            diagonal,
            filename: filename.to_string(),
            pixel_sensor,
            scale,
            max_sample_luminance,
        }))
    }

    pub fn new_gbuffer(
        resolution: &Point2i,
        pixel_bounds: &Bounds2i,
        filter: &Filter,
        diagonal: Float,
        filename: &str,
        scale: Float,
        max_sample_luminance: Float,
        gbuffer_coordinate_system: GBufferCoordinateSystem,
        pixel_sensor: PixelSensor,
    ) -> Self {
        Self::GBuffer(GBufferFilm::new(
            FilmBaseParameters {
                full_resolution: *resolution,
                pixel_bounds: *pixel_bounds,
                filter: filter.clone(),
                diagonal,
                filename: filename.to_string(),
                pixel_sensor,
                scale,
                max_sample_luminance,
            },
            gbuffer_coordinate_system,
        ))
    }

    pub fn new_spectral(
        resolution: &Point2i,
        pixel_bounds: &Bounds2i,
        filter: &Filter,
        diagonal: Float,
        filename: &str,
        scale: Float,
        max_sample_luminance: Float,
        pixel_sensor: PixelSensor,
    ) -> Self {
        Self::Spectral(SpectralFilm::new(FilmBaseParameters {
            full_resolution: *resolution,
            pixel_bounds: *pixel_bounds,
            filter: filter.clone(),
            diagonal,
            filename: filename.to_string(),
            pixel_sensor,
            scale,
            max_sample_luminance,
        }))
    }

    pub fn base(&self) -> &FilmBase {
        match self {
            Self::Rgb(film) => film.base(),
            Self::GBuffer(film) => film.base(),
            Self::Spectral(film) => film.base(),
        }
    }

    pub fn base_mut(&mut self) -> &mut FilmBase {
        match self {
            Self::Rgb(film) => film.base_mut(),
            Self::GBuffer(film) => film.base_mut(),
            Self::Spectral(film) => film.base_mut(),
        }
    }

    pub fn full_resolution(&self) -> Point2i {
        self.base().full_resolution()
    }

    pub fn diagonal(&self) -> Float {
        self.base().diagonal()
    }

    pub fn filename(&self) -> &str {
        self.base().filename()
    }

    pub fn cropped_pixel_bounds(&self) -> Bounds2i {
        self.base().cropped_pixel_bounds()
    }

    pub fn uses_visible_surface(&self) -> bool {
        match self {
            Self::Rgb(film) => film.uses_visible_surface(),
            Self::GBuffer(film) => film.uses_visible_surface(),
            Self::Spectral(film) => film.uses_visible_surface(),
        }
    }

    pub fn gbuffer_coordinate_system(&self) -> GBufferCoordinateSystem {
        match self {
            Self::GBuffer(film) => film.gbuffer_coordinate_system(),
            Self::Rgb(_) | Self::Spectral(_) => GBufferCoordinateSystem::Camera,
        }
    }

    /// Equivalent to pbrt-v4's `Film::SampleBounds()`.
    pub fn sample_bounds(&self) -> Bounds2i {
        self.base().sample_bounds()
    }

    pub fn get_sample_bounds(&self) -> Bounds2i {
        self.sample_bounds()
    }

    /// Equivalent to pbrt-v4's `Film::PixelBounds()`.
    pub fn pixel_bounds(&self) -> Bounds2i {
        self.base().pixel_bounds()
    }

    pub fn get_pixel_bounds(&self) -> Bounds2i {
        self.pixel_bounds()
    }

    /// Equivalent to pbrt-v4's `Film::GetFilter()` (Rust naming drops `get_`).
    pub fn filter(&self) -> &Filter {
        self.base().filter()
    }

    pub fn get_physical_extent(&self) -> Bounds2f {
        self.base().get_physical_extent()
    }

    pub fn get_film_tile(&self, sample_bounds: &Bounds2i) -> FilmTile {
        match self {
            Self::Rgb(film) => film.get_film_tile(sample_bounds),
            Self::GBuffer(film) => film.get_film_tile(sample_bounds),
            Self::Spectral(film) => film.get_film_tile(sample_bounds),
        }
    }

    pub fn sample_wavelengths(&self, u: Float) -> SampledWavelengths {
        match self {
            // pbrt-v4 dispatches SampleWavelengths through Film's
            // TaggedPointer; the spectral variant samples uniformly across
            // [lambda_min, lambda_max] instead of using the visible
            // importance distribution.
            Self::Spectral(film) => film.sample_wavelengths(u),
            Self::Rgb(_) | Self::GBuffer(_) => self.base().sample_wavelengths(u),
        }
    }

    pub fn merge_film_tile(&mut self, tile: &FilmTile) {
        let _p = ProfilePhase::new(Prof::MergeFilmTile);
        match self {
            Self::Rgb(film) => film.merge_film_tile(tile),
            Self::GBuffer(film) => film.merge_film_tile(tile),
            Self::Spectral(film) => film.merge_film_tile(tile),
        }
    }

    pub fn merge_splats(&self, splat_scale: Float) {
        match self {
            Self::Rgb(film) => film.merge_splats(splat_scale),
            Self::GBuffer(film) => film.merge_splats(splat_scale),
            Self::Spectral(film) => film.merge_splats(splat_scale),
        }
    }

    pub fn update_display(&self, bounds: &Bounds2i) {
        match self {
            Self::Rgb(film) => film.update_display(bounds),
            Self::GBuffer(film) => film.update_display(bounds),
            Self::Spectral(film) => film.update_display(bounds),
        }
    }

    pub fn update_display_scale(&self, bounds: &Bounds2i, scale: Float) {
        match self {
            Self::Rgb(film) => film.update_display_scale(bounds, scale),
            Self::GBuffer(film) => film.update_display_scale(bounds, scale),
            Self::Spectral(film) => film.update_display_scale(bounds, scale),
        }
    }

    pub fn set_image(&mut self, img: &[Spectrum]) {
        match self {
            Self::Rgb(film) => film.set_image(img),
            Self::GBuffer(film) => film.set_image(img),
            Self::Spectral(film) => film.set_image(img),
        }
    }

    pub fn add_splat(&mut self, p: &Vector2f, v: &Spectrum) {
        match self {
            Self::Rgb(film) => film.add_splat(p, v),
            Self::GBuffer(film) => film.add_splat(p, v),
            Self::Spectral(film) => film.add_splat(p, v),
        }
    }

    pub fn add_splat_with_wavelengths(
        &mut self,
        p: &Vector2f,
        v: &Spectrum,
        lambda: &SampledWavelengths,
    ) {
        match self {
            Self::Rgb(film) => film.add_splat_with_wavelengths(p, v, Some(lambda)),
            Self::GBuffer(film) => film.add_splat_with_wavelengths(p, v, Some(lambda)),
            Self::Spectral(film) => film.add_splat_with_wavelengths(p, v, Some(lambda)),
        }
    }

    /// pbrt-v4 `Film::AddSplat(Point2f, SampledSpectrum, SampledWavelengths)`.
    /// `&self` because the per-variant implementation goes through
    /// per-tile RwLocks; callers can hold `film.read()` and splat in
    /// parallel.
    pub fn add_splat_packet(&self, p: &Vector2f, v: &SampledSpectrum, lambda: &SampledWavelengths) {
        match self {
            Self::Rgb(film) => film.add_splat_packet(p, v, lambda),
            Self::GBuffer(film) => film.add_splat_packet(p, v, lambda),
            Self::Spectral(film) => film.add_splat_packet(p, v, lambda),
        }
    }

    /// SPPM-style direct RGB pixel write. Used by integrators that
    /// compute output radiance directly in RGB space (e.g.
    /// `SPPMIntegrator` converts SampledSpectrum to RGB per step via
    /// the pixel sensor and accumulates RGB across iterations).
    pub fn add_pixel_rgb(&self, p_pixel: Point2i, rgb: [Float; 3], weight: Float) {
        match self {
            Self::Rgb(film) => film.add_pixel_rgb(p_pixel, rgb, weight),
            Self::GBuffer(film) => film.add_pixel_rgb(p_pixel, rgb, weight),
            Self::Spectral(film) => film.add_pixel_rgb(p_pixel, rgb, weight),
        }
    }

    pub fn clear(&mut self) {
        match self {
            Self::Rgb(film) => film.clear(),
            Self::GBuffer(film) => film.clear(),
            Self::Spectral(film) => film.clear(),
        }
    }

    pub fn render_start(&self) {
        self.base().render_start();
    }

    pub fn render_end(&self) {
        self.base().render_end();
    }

    pub fn write_image(&self) {
        match self {
            Self::Rgb(film) => film.write_image(),
            Self::GBuffer(film) => film.write_image(),
            Self::Spectral(film) => film.write_image(),
        }
    }

    pub fn to_image(&self) -> Result<Image, PbrtError> {
        match self {
            Self::Rgb(film) => Ok(film.to_image()),
            Self::GBuffer(film) => Ok(film.to_image()),
            Self::Spectral(film) => Ok(film.to_image()),
        }
    }

    pub fn add_display(&self, display: &Arc<RwLock<dyn Display>>) {
        self.base().add_display(display);
    }
}

unsafe impl Send for Film {}
unsafe impl Sync for Film {}
