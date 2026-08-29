use super::film_base::{
    add_splat_into_pixels, add_splat_packet_into_pixels, normalize_pixel, FilmBase,
    FilmBaseParameters,
};
use super::film_tile::{FilmTile, SpectralTileConfig};
use super::splat_tile::*;
use crate::displays::DisplayTile;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::imageio::*;
use crate::util::spectrum::*;
use crate::util::AtomicDouble;

use log::*;
use std::sync::Mutex;

/// Default spectral range. Matches pbrt-v4 `SpectralFilm::Create` defaults
/// (`src/pbrt/film.cpp`).
pub const SPECTRAL_LAMBDA_MIN_DEFAULT: Float = 360.0;
pub const SPECTRAL_LAMBDA_MAX_DEFAULT: Float = 830.0;
pub const SPECTRAL_NUM_BUCKETS_DEFAULT: usize = 16;

/// Per-pixel storage for `SpectralFilm`. Mirrors pbrt-v4's
#[derive(Debug, Clone)]
pub struct SpectralPixel {
    pub rgb_sum: [Float; 3],
    pub filter_weight_sum: Float,
    pub bucket_sums: Vec<Float>,
    pub weight_sums: Vec<Float>,
}

impl SpectralPixel {
    pub fn zero(n_buckets: usize) -> Self {
        Self {
            rgb_sum: [0.0; 3],
            filter_weight_sum: 0.0,
            bucket_sums: vec![0.0; n_buckets],
            weight_sums: vec![0.0; n_buckets],
        }
    }
}

pub struct SpectralFilm {
    base: FilmBase,
    pixels: Mutex<Vec<SpectralPixel>>,
    splat_pixels: Vec<[AtomicDouble; 3]>,
    splat_scale: AtomicDouble,
    scale: Float,
    max_sample_luminance: Float,
    lambda_min: Float,
    lambda_max: Float,
    n_buckets: usize,
}

impl SpectralFilm {
    pub fn new(p: FilmBaseParameters) -> Self {
        Self::with_spectral_range(
            p,
            SPECTRAL_LAMBDA_MIN_DEFAULT,
            SPECTRAL_LAMBDA_MAX_DEFAULT,
            SPECTRAL_NUM_BUCKETS_DEFAULT,
        )
    }

    pub fn with_spectral_range(
        p: FilmBaseParameters,
        lambda_min: Float,
        lambda_max: Float,
        n_buckets: usize,
    ) -> Self {
        let pixel_bounds = p.pixel_bounds;
        let scale = p.scale;
        let max_sample_luminance = p.max_sample_luminance;
        let n_buckets = n_buckets.max(1);

        let area = pixel_bounds.area() as usize;
        let pixels: Vec<SpectralPixel> =
            (0..area).map(|_| SpectralPixel::zero(n_buckets)).collect();
        let splat_pixels = new_atomic_rgb_buffer(area);

        let base = FilmBase::new(&p);
        Self {
            base,
            pixels: Mutex::new(pixels),
            splat_pixels,
            splat_scale: AtomicDouble::new(1.0),
            scale,
            max_sample_luminance,
            lambda_min,
            lambda_max,
            n_buckets,
        }
    }

    pub fn base(&self) -> &FilmBase {
        &self.base
    }

    pub fn base_mut(&mut self) -> &mut FilmBase {
        &mut self.base
    }

    pub fn uses_visible_surface(&self) -> bool {
        false
    }

    pub fn lambda_min(&self) -> Float {
        self.lambda_min
    }

    pub fn lambda_max(&self) -> Float {
        self.lambda_max
    }

    pub fn n_buckets(&self) -> usize {
        self.n_buckets
    }

    /// Equivalent to pbrt-v4 `SpectralFilm::SampleWavelengths`: spectral
    /// films sample uniformly over `[lambda_min, lambda_max]` rather than
    /// visible-importance.
    pub fn sample_wavelengths(&self, u: Float) -> SampledWavelengths {
        SampledWavelengths::sample_uniform(u, self.lambda_min, self.lambda_max)
    }

    /// Equivalent to pbrt-v4 `SpectralFilm::LambdaToBucket`.
    pub fn lambda_to_bucket(&self, lambda: Float) -> usize {
        let span = self.lambda_max - self.lambda_min;
        let bucket = self.n_buckets as Float * (lambda - self.lambda_min) / span;
        (bucket as i32).clamp(0, self.n_buckets as i32 - 1) as usize
    }

    pub fn spectral_tile_config(&self) -> SpectralTileConfig {
        SpectralTileConfig {
            lambda_min: self.lambda_min,
            lambda_max: self.lambda_max,
            n_buckets: self.n_buckets,
        }
    }

    pub fn get_film_tile(&self, sample_bounds: &Bounds2i) -> FilmTile {
        self.base.make_film_tile_with_spectral(
            sample_bounds,
            self.max_sample_luminance,
            Some(self.spectral_tile_config()),
        )
    }

    pub fn merge_film_tile(&mut self, tile: &FilmTile) {
        // Pull both the RGB preview and spectral bucket output.
        // and the per-bucket spectral accumulators that the tile gathered
        // from `add_sample_filter*`. Mirrors pbrt-v4 SpectralFilm::AddSample
        // landing into `pixel.bucketSums[b]` / `weightSums[b]`.
        let bounds = tile.get_pixel_bounds();
        let n_buckets = self.n_buckets;
        let mut pixels = self.pixels.lock().unwrap();
        let has_spectral =
            tile.spectral.is_some() && tile.spectral_buckets.len() == tile.pixels.len() * n_buckets;
        for y in bounds.min.y..bounds.max.y {
            for x in bounds.min.x..bounds.max.x {
                let p = Vector2i::from((x, y));
                let src_index = tile.get_pixel_index(&p);
                let dst_index = self.base.get_pixel_index(&p);
                let src = &tile.pixels[src_index];
                for i in 0..3 {
                    pixels[dst_index].rgb_sum[i] += src.contrib_sum[i];
                }
                pixels[dst_index].filter_weight_sum += src.filter_weight_sum;
                if has_spectral {
                    let src_off = src_index * n_buckets;
                    let dst = &mut pixels[dst_index];
                    for b in 0..n_buckets {
                        dst.bucket_sums[b] += tile.spectral_buckets[src_off + b];
                        dst.weight_sums[b] += tile.spectral_bucket_weights[src_off + b];
                    }
                }
            }
        }
    }

    pub fn merge_splats(&self, splat_scale: Float) {
        self.splat_scale
            .store(splat_scale as f64, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn add_splat(&mut self, p: &Vector2f, v: &Spectrum) {
        self.add_splat_with_wavelengths(p, v, None);
    }

    /// Direct RGB pixel write for SPPM-style integrators (see
    /// `RGBFilm::add_pixel_rgb`).
    pub fn add_pixel_rgb(&self, p_pixel: Point2i, rgb: [Float; 3], weight: Float) {
        let bounds = self.base.cropped_pixel_bounds();
        if !bounds.inside_exclusive(&p_pixel) {
            return;
        }
        let pi = Point2i::new(p_pixel.x - bounds.min.x, p_pixel.y - bounds.min.y);
        let width = (bounds.max.x - bounds.min.x) as usize;
        let idx = pi.y as usize * width + pi.x as usize;
        let mut pixels = self.pixels.lock().unwrap();
        pixels[idx].rgb_sum[0] += rgb[0] * weight;
        pixels[idx].rgb_sum[1] += rgb[1] * weight;
        pixels[idx].rgb_sum[2] += rgb[2] * weight;
        pixels[idx].filter_weight_sum += weight;
    }

    pub fn add_splat_with_wavelengths(
        &mut self,
        p: &Vector2f,
        v: &Spectrum,
        lambda: Option<&SampledWavelengths>,
    ) {
        add_splat_into_pixels(
            &self.splat_pixels,
            self.base.cropped_pixel_bounds(),
            self.base.pixel_sensor_ref(),
            p,
            v,
            lambda,
        );
    }

    /// pbrt-v4 `Film::AddSplat(Point2f, SampledSpectrum, SampledWavelengths)`.
    /// `&self` because the per-pixel splat accumulators are atomic.
    pub fn add_splat_packet(&self, p: &Vector2f, v: &SampledSpectrum, lambda: &SampledWavelengths) {
        add_splat_packet_into_pixels(
            &self.splat_pixels,
            self.base.cropped_pixel_bounds(),
            self.base.pixel_sensor_ref(),
            self.base.filter(),
            self.max_sample_luminance,
            p,
            v,
            lambda,
        );
    }

    pub fn set_image(&mut self, img: &[Spectrum]) {
        let pixel_bounds = self.base.cropped_pixel_bounds();
        let sensor = self.base.pixel_sensor();
        let mut pixels = self.pixels.lock().unwrap();
        let n_pixels = pixel_bounds.area() as usize;
        if n_pixels <= img.len() {
            for i in 0..n_pixels {
                pixels[i].rgb_sum = sensor.to_output_rgb(&img[i]);
                pixels[i].filter_weight_sum = 1.0;
            }
        }
    }

    pub fn clear(&mut self) {
        let n_buckets = self.n_buckets;
        let mut pixels = self.pixels.lock().unwrap();
        for pixel in &mut *pixels {
            *pixel = SpectralPixel::zero(n_buckets);
        }
        clear_atomic_rgb_buffer(&self.splat_pixels);
        self.splat_scale
            .store(1.0, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn update_display(&self, bounds: &Bounds2i) {
        self.update_display_scale(bounds, 1.0);
    }

    pub fn update_display_scale(&self, bounds: &Bounds2i, scale: Float) {
        if self.base.display_is_empty() {
            return;
        }

        let scale = self.scale * scale;
        let cropped = self.base.cropped_pixel_bounds();
        let tx0 = i32::max(cropped.min.x, bounds.min.x) as usize;
        let ty0 = i32::max(cropped.min.y, bounds.min.y) as usize;
        let tx1 = i32::min(cropped.max.x, bounds.max.x) as usize;
        let ty1 = i32::min(cropped.max.y, bounds.max.y) as usize;

        if tx0 >= tx1 || ty0 >= ty1 {
            return;
        }

        let twidth = tx1 - tx0;
        let theight = ty1 - ty0;
        let mut buffer = vec![0.0; 3 * twidth * theight];
        let pixels = self.pixels.lock().unwrap();
        let splat_scale = self.splat_scale.load(std::sync::atomic::Ordering::Relaxed) as Float;
        for y in ty0..ty1 {
            for x in tx0..tx1 {
                let p = Vector2i::from((x as i32, y as i32));
                let src_index = self.base.get_pixel_index(&p);
                let c = normalize_pixel(
                    pixels[src_index].rgb_sum,
                    pixels[src_index].filter_weight_sum,
                    &load_atomic_rgb(&self.splat_pixels[src_index]),
                    scale * splat_scale,
                );
                let by = y - ty0;
                let bx = x - tx0;
                let idx = by * twidth + bx;
                buffer[3 * idx] = c[0] as f32;
                buffer[3 * idx + 1] = c[1] as f32;
                buffer[3 * idx + 2] = c[2] as f32;
            }
        }

        let display_tile = DisplayTile {
            x: tx0,
            y: ty0,
            width: twidth,
            height: theight,
            buffer,
        };
        self.base.push_display_tile(display_tile);
    }

    pub fn write_image(&self) {
        info!("Converting image to RGB + spectral channels for SpectralFilm output");
        let pixel_bounds = self.base.cropped_pixel_bounds();
        let area = pixel_bounds.area() as usize;
        let n_buckets = self.n_buckets;
        let pixels = self.pixels.lock().unwrap();
        let splat_scale = self.splat_scale.load(std::sync::atomic::Ordering::Relaxed) as Float;

        let mut r = vec![0.0f32; area];
        let mut g = vec![0.0f32; area];
        let mut b = vec![0.0f32; area];
        let mut bucket_channels: Vec<Vec<f32>> =
            (0..n_buckets).map(|_| vec![0.0f32; area]).collect();

        for offset in 0..pixels.len() {
            let c = normalize_pixel(
                pixels[offset].rgb_sum,
                pixels[offset].filter_weight_sum,
                &load_atomic_rgb(&self.splat_pixels[offset]),
                self.scale * splat_scale,
            );
            r[offset] = c[0] as f32;
            g[offset] = c[1] as f32;
            b[offset] = c[2] as f32;

            let px = &pixels[offset];
            for (idx, ch) in bucket_channels.iter_mut().enumerate() {
                let w = px.weight_sums[idx];
                if w > 0.0 {
                    ch[offset] = (px.bucket_sums[idx] / w) as f32;
                }
            }
        }

        let channel_names: Vec<String> = (0..n_buckets)
            .map(|i| {
                // Bucket center wavelength, formatted "%.3fnm" with '.' -> ','
                // to match pbrt-v4 SpectralFilm::GetImage. OpenEXR uses '.'
                // as the layer separator, so the comma keeps the lambda
                // grouped under the same "S0.<lambda>nm" channel name.
                let t = (i as Float + 0.5) / n_buckets as Float;
                let lambda = self.lambda_min + t * (self.lambda_max - self.lambda_min);
                let s = format!("{:.3}nm", lambda).replace('.', ",");
                format!("S0.{}", s)
            })
            .collect();

        let mut channels: Vec<(&str, Vec<f32>)> = Vec::with_capacity(3 + n_buckets);
        channels.push(("R", r));
        channels.push(("G", g));
        channels.push(("B", b));
        for (name, samples) in channel_names.iter().zip(bucket_channels.into_iter()) {
            channels.push((name.as_str(), samples));
        }

        info!(
            "Writing image {} with bounds {:?} ({} spectral channels)",
            self.base.filename(),
            pixel_bounds,
            n_buckets
        );
        // Match pbrt-v4 film.cpp:1021-1025 metadata.
        let attrs: &[(&str, &str)] = &[
            ("spectralLayoutVersion", "1.0"),
            ("emissiveUnits", "W.m^-2.sr^-1"),
        ];
        let total_resolution = self.base.full_resolution();
        let _ = write_image_exr_channels_with_attrs(
            self.base.filename(),
            &pixel_bounds,
            &total_resolution,
            &channels,
            attrs,
        );
    }

    pub fn to_image(&self) -> crate::util::image::Image {
        let pixel_bounds = self.base.cropped_pixel_bounds();
        let area = pixel_bounds.area() as usize;
        let pixels = self.pixels.lock().unwrap();
        let splat_scale = self.splat_scale.load(std::sync::atomic::Ordering::Relaxed) as Float;
        let mut texels = Vec::with_capacity(area);
        for i in 0..area {
            let rgb = normalize_pixel(
                pixels[i].rgb_sum,
                pixels[i].filter_weight_sum,
                &load_atomic_rgb(&self.splat_pixels[i]),
                self.scale * splat_scale,
            );
            texels.push(RGBSpectrum::new(rgb[0], rgb[1], rgb[2]));
        }
        crate::util::image::Image::new(pixel_bounds.diagonal(), texels)
    }
}
