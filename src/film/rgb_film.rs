use super::film_base::{
    add_splat_into_pixels, add_splat_packet_into_pixels, normalize_pixel, FilmBase,
    FilmBaseParameters,
};
use super::film_tile::FilmTile;
use super::splat_tile::*;
use crate::displays::DisplayTile;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::image::Image;
use crate::util::imageio::*;
use crate::util::spectrum::*;
use crate::util::AtomicDouble;

use log::*;
use std::sync::Mutex;

/// Per-pixel storage for `RGBFilm`. Mirrors pbrt-v4's
/// `RGBFilm::Pixel { rgbSum[3], weightSum, rgbSplat[3] }` (see
/// src/pbrt/film.h). The splat channel is held in a separate per-pixel
/// `AtomicDouble` buffer, matching pbrt-v4's `rgbSplat` storage.
#[derive(Debug, Default, Copy, Clone)]
pub struct RGBPixel {
    pub rgb_sum: [Float; 3],
    pub filter_weight_sum: Float,
}

impl RGBPixel {
    pub fn zero() -> Self {
        Self {
            rgb_sum: [0.0; 3],
            filter_weight_sum: 0.0,
        }
    }
}

pub struct RGBFilm {
    base: FilmBase,
    pixels: Mutex<Vec<RGBPixel>>,
    splat_pixels: Vec<[AtomicDouble; 3]>,
    splat_scale: AtomicDouble,
    scale: Float,
    max_sample_luminance: Float,
}

impl RGBFilm {
    pub fn new(p: FilmBaseParameters) -> Self {
        let pixel_bounds = p.pixel_bounds;
        let scale = p.scale;
        let max_sample_luminance = p.max_sample_luminance;

        let pixels = vec![RGBPixel::zero(); pixel_bounds.area() as usize];
        let splat_pixels = new_atomic_rgb_buffer(pixel_bounds.area() as usize);

        let base = FilmBase::new(&p);
        Self {
            base,
            pixels: Mutex::new(pixels),
            splat_pixels,
            splat_scale: AtomicDouble::new(1.0),
            scale,
            max_sample_luminance,
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

    pub fn get_film_tile(&self, sample_bounds: &Bounds2i) -> FilmTile {
        self.base
            .make_film_tile(sample_bounds, self.max_sample_luminance)
    }

    pub fn merge_film_tile(&mut self, tile: &FilmTile) {
        let bounds = tile.get_pixel_bounds();
        let mut pixels = self.pixels.lock().unwrap();
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

    /// pbrt-v4 SPPM writes its per-pixel result directly as RGB
    /// (bypassing the pixel sensor / sampled spectrum path) since L is
    /// accumulated in RGB throughout the algorithm. This method writes
    /// `rgb` into the pixel's rgb_sum with `weight` accumulated into
    /// filter_weight_sum, so a subsequent normalize_pixel produces the
    /// raw RGB.
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
        let mut pixels = self.pixels.lock().unwrap();
        for pixel in &mut *pixels {
            *pixel = RGBPixel::zero();
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
        info!("Converting image to RGB and computing final weighted pixel values");
        let (pixel_bounds, rgb) = self.final_rgb_pixels();
        info!(
            "Writing image {} with bounds {:?}",
            self.base.filename(),
            pixel_bounds
        );
        let _ = write_image(
            self.base.filename(),
            &rgb,
            &pixel_bounds,
            &self.base.full_resolution(),
        );
    }

    pub fn to_image(&self) -> Image {
        let (pixel_bounds, rgb) = self.final_rgb_pixels();
        let resolution = Point2i::new(
            pixel_bounds.max.x - pixel_bounds.min.x,
            pixel_bounds.max.y - pixel_bounds.min.y,
        );
        let texels = rgb
            .chunks_exact(3)
            .map(|c| RGBSpectrum::new(c[0], c[1], c[2]))
            .collect();
        Image::new(resolution, texels)
    }

    fn final_rgb_pixels(&self) -> (Bounds2i, Vec<Float>) {
        let pixel_bounds = self.base.cropped_pixel_bounds();
        let area = pixel_bounds.area() as usize;
        let mut rgb = vec![0.0; 3 * area];
        let pixels = self.pixels.lock().unwrap();
        let splat_scale = self.splat_scale.load(std::sync::atomic::Ordering::Relaxed) as Float;
        for offset in 0..pixels.len() {
            let c = normalize_pixel(
                pixels[offset].rgb_sum,
                pixels[offset].filter_weight_sum,
                &load_atomic_rgb(&self.splat_pixels[offset]),
                self.scale * splat_scale,
            );
            rgb[3 * offset] = c[0];
            rgb[3 * offset + 1] = c[1];
            rgb[3 * offset + 2] = c[2];
        }
        (pixel_bounds, rgb)
    }
}
