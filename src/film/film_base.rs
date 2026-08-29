use super::film_tile::*;
use super::pixel_sensor::PixelSensor;
use super::splat_tile::{add_atomic_rgb, AtomicRgb};
use crate::base::filter::Filter;
use crate::displays::MultipleDisplay;
use crate::displays::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::spectrum::*;

use log::*;
use std::sync::{Arc, RwLock};

/// Bundle of parameters shared by all Film variants, matching pbrt-v4's
/// `FilmBaseParameters` in `src/pbrt/film.h`. `scale` and
/// `max_sample_luminance` are r4-only knobs that live on each variant
/// (mirroring pbrt-v4 where `maxComponentValue` is RGBFilm-specific).
pub struct FilmBaseParameters {
    pub full_resolution: Point2i,
    pub pixel_bounds: Bounds2i,
    pub filter: Filter,
    pub diagonal: Float,
    pub filename: String,
    pub pixel_sensor: PixelSensor,
    pub scale: Float,
    pub max_sample_luminance: Float,
}

/// `FilmBase` matches pbrt-v4's `FilmBase` (src/pbrt/film.h): it carries
/// only the geometry/metadata fields that every Film variant needs. The
/// per-variant pixel storage lives in `RGBFilm`, `GBufferFilm`, and
/// `SpectralFilm` themselves, mirroring v4's `Array2D<Pixel>` members.
///
/// `filter_table` and `display` are r4 extensions. `filter_table` is a
/// precomputed lookup over `filter` (an r4 perf optimization); `display` is
/// the live-preview hook. Both stay on FilmBase because they are shared by
/// every variant and derive from fields already living here.
pub struct FilmBase {
    full_resolution: Point2i,
    diagonal: Float,
    filter: Filter,
    filename: String,
    cropped_pixel_bounds: Bounds2i,
    pixel_sensor: PixelSensor,
    filter_table: Arc<[Float; FT_SZ]>,
    display: MultipleDisplay,
}

impl FilmBase {
    pub fn new(p: &FilmBaseParameters) -> Self {
        info!(
            "Created film with full resolution {:?}, pixel bounds {:?}",
            p.full_resolution, p.pixel_bounds
        );

        let mut filter_table: [Float; FT_SZ] = [1.0; FT_SZ];
        let radius = p.filter.radius();
        for y in 0..FT_W {
            for x in 0..FT_W {
                let xx = (x as Float) * (radius.x / (FT_W - 1) as Float);
                let yy = (y as Float) * (radius.y / (FT_W - 1) as Float);
                filter_table[y * FT_W + x] = p.filter.evaluate(&Vector2f::new(xx, yy));
            }
        }

        Self {
            full_resolution: p.full_resolution,
            diagonal: 0.001 * p.diagonal,
            filter: p.filter.clone(),
            filename: p.filename.clone(),
            cropped_pixel_bounds: p.pixel_bounds,
            pixel_sensor: p.pixel_sensor.clone(),
            filter_table: Arc::new(filter_table),
            display: MultipleDisplay::new(),
        }
    }

    pub fn full_resolution(&self) -> Point2i {
        self.full_resolution
    }

    pub fn diagonal(&self) -> Float {
        self.diagonal
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn pixel_sensor(&self) -> PixelSensor {
        self.pixel_sensor.clone()
    }

    pub fn pixel_sensor_ref(&self) -> &PixelSensor {
        &self.pixel_sensor
    }

    pub fn sample_wavelengths(&self, u: Float) -> SampledWavelengths {
        SampledWavelengths::sample_visible(u)
    }

    pub fn cropped_pixel_bounds(&self) -> Bounds2i {
        self.cropped_pixel_bounds
    }

    /// Equivalent to pbrt-v4's `Film::PixelBounds()`.
    /// `cropped_pixel_bounds()` / `get_pixel_bounds()` accessors are aliases
    /// kept while callers migrate.
    pub fn pixel_bounds(&self) -> Bounds2i {
        self.cropped_pixel_bounds
    }

    /// Equivalent to pbrt-v4's `Film::SampleBounds()`.
    /// `get_sample_bounds()` accessor is an alias kept while callers migrate.
    pub fn sample_bounds(&self) -> Bounds2i {
        let radius = self.filter.radius();
        let p_min = self.cropped_pixel_bounds.min;
        let p_max = self.cropped_pixel_bounds.max;
        let x0 = Float::floor(p_min.x as Float - radius.x + 0.5) as i32;
        let y0 = Float::floor(p_min.y as Float - radius.y + 0.5) as i32;
        let x1 = Float::ceil(p_max.x as Float + radius.x - 0.5) as i32;
        let y1 = Float::ceil(p_max.y as Float + radius.y - 0.5) as i32;
        Bounds2i::new(&Vector2i::new(x0, y0), &Vector2i::new(x1, y1))
    }

    pub fn get_sample_bounds(&self) -> Bounds2i {
        self.sample_bounds()
    }

    pub fn get_pixel_bounds(&self) -> Bounds2i {
        self.pixel_bounds()
    }

    /// Equivalent to pbrt-v4's `Film::GetFilter()`. Rust naming convention
    /// drops the `get_` prefix.
    pub fn filter(&self) -> &Filter {
        &self.filter
    }

    pub fn get_physical_extent(&self) -> Bounds2f {
        let aspect = self.full_resolution.y as Float / self.full_resolution.x as Float;
        let x = Float::sqrt(self.diagonal * self.diagonal / (1.0 + aspect * aspect));
        let y = aspect * x;
        Bounds2f::new(
            &Point2f::new(-x / 2.0, -y / 2.0),
            &Point2f::new(x / 2.0, y / 2.0),
        )
    }

    pub fn get_pixel_index(&self, p: &Vector2i) -> usize {
        let cmaxx = self.cropped_pixel_bounds.max.x as usize;
        let cminx = self.cropped_pixel_bounds.min.x as usize;
        let cminy = self.cropped_pixel_bounds.min.y as usize;
        let x = p.x as usize;
        let y = p.y as usize;
        let width = cmaxx - cminx;
        (x - cminx) + (y - cminy) * width
    }

    /// Build a `FilmTile` for the given sample bounds. Variants pass their
    /// own `max_sample_luminance` here so each variant can keep its own
    /// clamp setting (matching pbrt-v4's RGBFilm/GBufferFilm/SpectralFilm
    /// each owning their `maxComponentValue`).
    pub fn make_film_tile(
        &self,
        sample_bounds: &Bounds2i,
        max_sample_luminance: Float,
    ) -> FilmTile {
        self.make_film_tile_with_spectral(sample_bounds, max_sample_luminance, None)
    }

    /// Same as `make_film_tile` but lets `SpectralFilm` plug in a
    /// `SpectralTileConfig` so the tile carries spectral bucket
    /// accumulators in addition to the RGB-projected contribution.
    pub fn make_film_tile_with_spectral(
        &self,
        sample_bounds: &Bounds2i,
        max_sample_luminance: Float,
        spectral: Option<SpectralTileConfig>,
    ) -> FilmTile {
        let radius = self.filter.radius();
        let i2f = |v: &Vector2i| Vector2f::new(v.x as Float, v.y as Float);
        let floor = |v: &Vector2f| Vector2i::new(v.x.floor() as i32, v.y.floor() as i32);
        let ceil = |v: &Vector2f| Vector2i::new(v.x.ceil() as i32, v.y.ceil() as i32);
        let p0 = floor(&(i2f(&sample_bounds.min) - radius));
        let p1 = ceil(&(i2f(&sample_bounds.max) + radius));
        let tile_pixel_bounds = Bounds2i::new(&p0, &p1).intersect(&self.cropped_pixel_bounds);
        FilmTile::with_spectral(
            &tile_pixel_bounds,
            &radius,
            &self.filter_table,
            max_sample_luminance,
            self.pixel_sensor.clone(),
            spectral,
        )
    }

    pub fn add_display(&self, display: &Arc<RwLock<dyn Display>>) {
        self.display.add_display(display);
    }

    pub fn render_start(&self) {
        let resolution = [
            self.full_resolution[0] as usize,
            self.full_resolution[1] as usize,
        ];
        let channel_names = ["R", "G", "B"];
        let filename = self.filename.clone();
        let _ = self.display.start(&filename, &resolution, &channel_names);
    }

    pub fn render_end(&self) {
        let _ = self.display.end();
    }

    pub fn display_is_empty(&self) -> bool {
        self.display.is_empty()
    }

    pub fn push_display_tile(&self, tile: DisplayTile) {
        self.display
            .update(&tile)
            .or_else(|e| -> Result<(), PbrtError> {
                warn!("{:?}", e);
                Ok(())
            })
            .unwrap();
    }
}

pub fn add_splat_into_pixels(
    splat_pixels: &[AtomicRgb],
    cropped_pixel_bounds: Bounds2i,
    pixel_sensor: &PixelSensor,
    p: &Vector2f,
    v: &Spectrum,
    lambda: Option<&SampledWavelengths>,
) {
    let pi = Point2i::new(p.x.floor() as i32, p.y.floor() as i32);
    if !cropped_pixel_bounds.inside_exclusive(&pi) {
        return;
    }

    let pi = Point2i::new(
        pi.x - cropped_pixel_bounds.min.x,
        pi.y - cropped_pixel_bounds.min.y,
    );
    let rgb = match lambda {
        Some(lambda) => pixel_sensor.to_output_rgb_with_wavelengths(v, lambda),
        None => pixel_sensor.to_output_rgb(v),
    };

    let index = pi.y as usize * (cropped_pixel_bounds.max.x - cropped_pixel_bounds.min.x) as usize
        + pi.x as usize;
    add_atomic_rgb(&splat_pixels[index], rgb);
}

/// Packet variant of `add_splat_into_pixels` -- takes a `SampledSpectrum`
/// directly. pbrt-v4 `Film::AddSplat(Point2f, SampledSpectrum, lambda)`
/// pathway. Used by integrators that splat per-wavelength radiance
/// (LightPathIntegrator, BDPT, etc.) without going through `Spectrum`.
pub fn add_splat_packet_into_pixels(
    splat_pixels: &[AtomicRgb],
    cropped_pixel_bounds: Bounds2i,
    pixel_sensor: &PixelSensor,
    filter: &Filter,
    max_sample_luminance: Float,
    p: &Vector2f,
    v: &SampledSpectrum,
    lambda: &SampledWavelengths,
) {
    let filter_integral = filter.integral();
    if filter_integral == 0.0 {
        return;
    }
    let mut sensor_rgb = pixel_sensor.to_sensor_rgb_from_packet(v, lambda);
    let m = sensor_rgb[0].max(sensor_rgb[1]).max(sensor_rgb[2]);
    if m > max_sample_luminance {
        let scale = max_sample_luminance / m;
        sensor_rgb[0] *= scale;
        sensor_rgb[1] *= scale;
        sensor_rgb[2] *= scale;
    }
    let rgb = pixel_sensor.apply_output_matrix(&sensor_rgb);

    let p_discrete = *p + Vector2f::new(0.5, 0.5);
    let radius = filter.radius();
    let p0 = Point2i::new(
        (p_discrete.x - radius.x).floor() as i32,
        (p_discrete.y - radius.y).floor() as i32,
    );
    let p1 = Point2i::new(
        (p_discrete.x + radius.x).floor() as i32 + 1,
        (p_discrete.y + radius.y).floor() as i32 + 1,
    );
    let splat_bounds = Bounds2i::new(&p0, &p1).intersect(&cropped_pixel_bounds);

    for pi in &splat_bounds {
        let d = Point2f::new(p.x - pi.x as Float - 0.5, p.y - pi.y as Float - 0.5);
        let wt = filter.evaluate(&d);
        if wt == 0.0 {
            continue;
        }
        let local = Point2i::new(
            pi.x - cropped_pixel_bounds.min.x,
            pi.y - cropped_pixel_bounds.min.y,
        );
        let width = (cropped_pixel_bounds.max.x - cropped_pixel_bounds.min.x) as usize;
        let index = local.y as usize * width + local.x as usize;
        let scale = wt / filter_integral;
        add_atomic_rgb(
            &splat_pixels[index],
            [scale * rgb[0], scale * rgb[1], scale * rgb[2]],
        );
    }
}

/// Combine a Pixel's running RGB sum with its splat contribution and an
/// output scale, mirroring v4's `GetPixelRGB` body (film.h:268). v4 does
/// **not** clamp negative output components — bright out-of-gamut samples
/// can drive `outputRGBFromSensorRGB * sensorRGB` to small negatives in
/// individual channels and pbrt-v4 stores those untouched in the EXR.
pub fn normalize_pixel(
    rgb_sum: [Float; 3],
    filter_weight_sum: Float,
    splat_pixel: &[Float; 3],
    scale: Float,
) -> [Float; 3] {
    let mut c = rgb_sum;
    if filter_weight_sum > 0.0 {
        let inv_wt = 1.0 / filter_weight_sum;
        c[0] *= inv_wt;
        c[1] *= inv_wt;
        c[2] *= inv_wt;
    }
    c[0] = (c[0] + splat_pixel[0]) * scale;
    c[1] = (c[1] + splat_pixel[1]) * scale;
    c[2] = (c[2] + splat_pixel[2]) * scale;
    c
}
