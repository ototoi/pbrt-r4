use super::film_base::{
    add_splat_into_tiles, add_splat_packet_into_tiles, make_splat_tiles, merge_splat_tiles,
    normalize_pixel, FilmBase, FilmBaseParameters, FILM_PIXEL_MEMORY,
};
use super::film_tile::FilmTile;
use super::splat_tile::*;
use crate::base::film::GBufferCoordinateSystem;
use crate::displays::DisplayTile;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::imageio::*;
use crate::util::profile::*;
use crate::util::spectrum::*;

use log::*;
use std::sync::{Arc, Mutex, RwLock};

/// Per-pixel storage for `GBufferFilm`. Mirrors pbrt-v4's
/// `GBufferFilm::Pixel` (src/pbrt/film.h), where the RGB accumulator and
/// the gbuffer fields share a single Pixel struct rather than being
/// stored in two parallel buffers. The splat channel still lives in
/// per-tile aggregation buffers for portability.
#[derive(Debug, Default, Copy, Clone)]
pub struct GBufferPixel {
    pub rgb_sum: [Float; 3],
    pub filter_weight_sum: Float,
    pub p_sum: [Float; 3],
    pub dzdx_sum: Float,
    pub dzdy_sum: Float,
    pub n_sum: [Float; 3],
    pub ns_sum: [Float; 3],
    pub uv_sum: [Float; 2],
    pub albedo_sum: [Float; 3],
    pub gbuffer_weight_sum: Float,
}

impl GBufferPixel {
    pub fn zero() -> Self {
        Self::default()
    }
}

pub fn normalize_gbuffer_normal(sum: [Float; 3]) -> [Float; 3] {
    let len2 = sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2];
    if len2 > 0.0 {
        let inv_len = 1.0 / len2.sqrt();
        [sum[0] * inv_len, sum[1] * inv_len, sum[2] * inv_len]
    } else {
        [0.0; 3]
    }
}

pub struct GBufferFilm {
    base: FilmBase,
    pixels: Mutex<Vec<GBufferPixel>>,
    splat_pixels: Mutex<Vec<[Float; 3]>>,
    splat_tiles: Vec<Arc<RwLock<SplatTile>>>,
    splat_size: Vector2i,
    scale: Float,
    max_sample_luminance: Float,
    gbuffer_coordinate_system: GBufferCoordinateSystem,
}

impl GBufferFilm {
    pub fn new(p: FilmBaseParameters, gbuffer_coordinate_system: GBufferCoordinateSystem) -> Self {
        let pixel_bounds = p.pixel_bounds;
        let scale = p.scale;
        let max_sample_luminance = p.max_sample_luminance;

        let pixels = vec![GBufferPixel::zero(); pixel_bounds.area() as usize];
        FILM_PIXEL_MEMORY.with(|s| {
            s.add(pixel_bounds.area() as usize * std::mem::size_of::<GBufferPixel>());
        });
        let splat_pixels = vec![[0.0; 3]; pixel_bounds.area() as usize];
        let (splat_tiles, splat_size) = make_splat_tiles(&pixel_bounds);

        let base = FilmBase::new(&p);
        Self {
            base,
            pixels: Mutex::new(pixels),
            splat_pixels: Mutex::new(splat_pixels),
            splat_tiles,
            splat_size,
            scale,
            max_sample_luminance,
            gbuffer_coordinate_system,
        }
    }

    pub fn base(&self) -> &FilmBase {
        &self.base
    }

    pub fn base_mut(&mut self) -> &mut FilmBase {
        &mut self.base
    }

    pub fn uses_visible_surface(&self) -> bool {
        true
    }

    pub fn gbuffer_coordinate_system(&self) -> GBufferCoordinateSystem {
        self.gbuffer_coordinate_system
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
                let dst = &mut pixels[dst_index];
                for i in 0..3 {
                    dst.rgb_sum[i] += src.contrib_sum[i];
                    dst.p_sum[i] += src.p_sum[i];
                    dst.n_sum[i] += src.n_sum[i];
                    dst.ns_sum[i] += src.ns_sum[i];
                    dst.albedo_sum[i] += src.albedo_sum[i];
                }
                dst.filter_weight_sum += src.filter_weight_sum;
                dst.dzdx_sum += src.dzdx_sum;
                dst.dzdy_sum += src.dzdy_sum;
                dst.uv_sum[0] += src.uv_sum[0];
                dst.uv_sum[1] += src.uv_sum[1];
                dst.gbuffer_weight_sum += src.gbuffer_weight_sum;
            }
        }
    }

    pub fn merge_splats(&self, splat_scale: Float) {
        let mut splat_pixels = self.splat_pixels.lock().unwrap();
        merge_splat_tiles(
            &self.splat_tiles,
            &mut splat_pixels,
            self.base.cropped_pixel_bounds(),
            splat_scale,
        );
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
        let _p = ProfilePhase::new(Prof::SplatFilm);
        add_splat_into_tiles(
            &self.splat_tiles,
            self.splat_size,
            self.base.cropped_pixel_bounds(),
            self.base.pixel_sensor_ref(),
            p,
            v,
            lambda,
        );
    }

    /// pbrt-v4 `Film::AddSplat(Point2f, SampledSpectrum, SampledWavelengths)`.
    /// `&self` because all mutation flows through the per-tile RwLock
    /// (see `RGBFilm::add_splat_packet`).
    pub fn add_splat_packet(&self, p: &Vector2f, v: &SampledSpectrum, lambda: &SampledWavelengths) {
        let _p = ProfilePhase::new(Prof::SplatFilm);
        add_splat_packet_into_tiles(
            &self.splat_tiles,
            self.splat_size,
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
            *pixel = GBufferPixel::zero();
        }
        let mut splat_pixels = self.splat_pixels.lock().unwrap();
        for pixel in &mut *splat_pixels {
            *pixel = [0.0; 3];
        }
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
        let splat_pixels = self.splat_pixels.lock().unwrap();
        for y in ty0..ty1 {
            for x in tx0..tx1 {
                let p = Vector2i::from((x as i32, y as i32));
                let src_index = self.base.get_pixel_index(&p);
                let c = normalize_pixel(
                    pixels[src_index].rgb_sum,
                    pixels[src_index].filter_weight_sum,
                    &splat_pixels[src_index],
                    scale,
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
        info!(
            "Converting gbuffer image {} with bounds {:?}",
            self.base.filename(),
            self.base.cropped_pixel_bounds()
        );

        let pixel_bounds = self.base.cropped_pixel_bounds();
        let area = pixel_bounds.area() as usize;
        let pixels = self.pixels.lock().unwrap();
        let splat_pixels = self.splat_pixels.lock().unwrap();

        let mut r = vec![0.0f32; area];
        let mut g = vec![0.0f32; area];
        let mut b = vec![0.0f32; area];
        let mut albedo_r = vec![0.0f32; area];
        let mut albedo_g = vec![0.0f32; area];
        let mut albedo_b = vec![0.0f32; area];
        let mut p_x = vec![0.0f32; area];
        let mut p_y = vec![0.0f32; area];
        let mut p_z = vec![0.0f32; area];
        let mut dzdx = vec![0.0f32; area];
        let mut dzdy = vec![0.0f32; area];
        let mut n_x = vec![0.0f32; area];
        let mut n_y = vec![0.0f32; area];
        let mut n_z = vec![0.0f32; area];
        let mut ns_x = vec![0.0f32; area];
        let mut ns_y = vec![0.0f32; area];
        let mut ns_z = vec![0.0f32; area];
        let mut u = vec![0.0f32; area];
        let mut v = vec![0.0f32; area];
        let zero = vec![0.0f32; area];

        for i in 0..area {
            let rgb = normalize_pixel(
                pixels[i].rgb_sum,
                pixels[i].filter_weight_sum,
                &splat_pixels[i],
                self.scale,
            );
            r[i] = rgb[0] as f32;
            g[i] = rgb[1] as f32;
            b[i] = rgb[2] as f32;

            if pixels[i].filter_weight_sum > 0.0 {
                let inv_wt = 1.0 / pixels[i].filter_weight_sum;
                albedo_r[i] = (pixels[i].albedo_sum[0] * inv_wt) as f32;
                albedo_g[i] = (pixels[i].albedo_sum[1] * inv_wt) as f32;
                albedo_b[i] = (pixels[i].albedo_sum[2] * inv_wt) as f32;
            }

            if pixels[i].gbuffer_weight_sum > 0.0 {
                let inv_wt = 1.0 / pixels[i].gbuffer_weight_sum;
                p_x[i] = (pixels[i].p_sum[0] * inv_wt) as f32;
                p_y[i] = (pixels[i].p_sum[1] * inv_wt) as f32;
                p_z[i] = (pixels[i].p_sum[2] * inv_wt) as f32;
                dzdx[i] = (pixels[i].dzdx_sum * inv_wt).abs() as f32;
                dzdy[i] = (pixels[i].dzdy_sum * inv_wt).abs() as f32;
                let n = normalize_gbuffer_normal(pixels[i].n_sum);
                let ns = normalize_gbuffer_normal(pixels[i].ns_sum);
                n_x[i] = n[0] as f32;
                n_y[i] = n[1] as f32;
                n_z[i] = n[2] as f32;
                ns_x[i] = ns[0] as f32;
                ns_y[i] = ns[1] as f32;
                ns_z[i] = ns[2] as f32;
                u[i] = (pixels[i].uv_sum[0] * inv_wt) as f32;
                v[i] = (pixels[i].uv_sum[1] * inv_wt) as f32;
            }
        }

        let channels = vec![
            ("R", r),
            ("G", g),
            ("B", b),
            ("Albedo.R", albedo_r),
            ("Albedo.G", albedo_g),
            ("Albedo.B", albedo_b),
            ("P.X", p_x),
            ("P.Y", p_y),
            ("P.Z", p_z),
            ("dzdx", dzdx),
            ("dzdy", dzdy),
            ("N.X", n_x),
            ("N.Y", n_y),
            ("N.Z", n_z),
            ("Ns.X", ns_x),
            ("Ns.Y", ns_y),
            ("Ns.Z", ns_z),
            ("u", u),
            ("v", v),
            ("Variance.R", zero.clone()),
            ("Variance.G", zero.clone()),
            ("Variance.B", zero.clone()),
            ("RelativeVariance.R", zero.clone()),
            ("RelativeVariance.G", zero.clone()),
            ("RelativeVariance.B", zero),
        ];
        let total_resolution = self.base.full_resolution();
        let meta = ExrExtraMeta::default();
        let _ = write_image_exr_channels_half(
            self.base.filename(),
            &pixel_bounds,
            &total_resolution,
            &channels,
            &meta,
        );
    }

    pub fn to_image(&self) -> crate::util::image::Image {
        let pixel_bounds = self.base.cropped_pixel_bounds();
        let area = pixel_bounds.area() as usize;
        let pixels = self.pixels.lock().unwrap();
        let splat_pixels = self.splat_pixels.lock().unwrap();
        let mut texels = Vec::with_capacity(area);
        for i in 0..area {
            let rgb = normalize_pixel(
                pixels[i].rgb_sum,
                pixels[i].filter_weight_sum,
                &splat_pixels[i],
                self.scale,
            );
            texels.push(RGBSpectrum::new(rgb[0], rgb[1], rgb[2]));
        }
        crate::util::image::Image::new(pixel_bounds.diagonal(), texels)
    }
}
