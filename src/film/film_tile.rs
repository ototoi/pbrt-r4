use super::pixel_sensor::PixelSensor;
use super::visible_surface::VisibleSurface;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::spectrum::*;

use std::sync::Arc;

pub const FILTER_TABLE_WIDTH: usize = 16;
pub const FT_W: usize = FILTER_TABLE_WIDTH;
pub const FT_SZ: usize = FILTER_TABLE_WIDTH * FILTER_TABLE_WIDTH;

fn gbuffer_albedo_to_rgb(albedo: SampledSpectrum, lambda: &SampledWavelengths) -> [Float; 3] {
    let mut illuminant = [0.0; SampledSpectrum::N_SAMPLES];
    for i in 0..SampledSpectrum::N_SAMPLES {
        illuminant[i] = d65_sample(lambda[i]);
    }
    (albedo * SampledSpectrum::from(illuminant)).to_rgb(lambda)
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct FilmTilePixel {
    pub contrib_sum: [Float; 3],
    pub filter_weight_sum: Float,
    pub gbuffer_weight_sum: Float,
    pub p_sum: [Float; 3],
    pub dzdx_sum: Float,
    pub dzdy_sum: Float,
    pub n_sum: [Float; 3],
    pub ns_sum: [Float; 3],
    pub uv_sum: [Float; 2],
    pub albedo_sum: [Float; 3],
}

impl FilmTilePixel {
    pub fn zero() -> Self {
        FilmTilePixel {
            contrib_sum: [0.0; 3],
            filter_weight_sum: 0.0,
            gbuffer_weight_sum: 0.0,
            p_sum: [0.0; 3],
            dzdx_sum: 0.0,
            dzdy_sum: 0.0,
            n_sum: [0.0; 3],
            ns_sum: [0.0; 3],
            uv_sum: [0.0; 2],
            albedo_sum: [0.0; 3],
        }
    }
}

/// Per-tile spectral configuration. Set by `SpectralFilm::get_film_tile`
/// so that `FilmTile::add_sample_*` can distribute each sample's
/// `SampledSpectrum` into the right wavelength buckets in addition to the
/// RGB-projected contribution. Other Film variants leave the spectral
/// path inactive (`spectral: None`).
#[derive(Debug, Clone, Copy)]
pub struct SpectralTileConfig {
    pub lambda_min: Float,
    pub lambda_max: Float,
    pub n_buckets: usize,
}

impl SpectralTileConfig {
    /// Same mapping as `SpectralFilm::lambda_to_bucket`, mirroring
    /// pbrt-v4 `SpectralFilm::LambdaToBucket`.
    pub fn lambda_to_bucket(&self, lambda: Float) -> usize {
        let span = self.lambda_max - self.lambda_min;
        let bucket = self.n_buckets as Float * (lambda - self.lambda_min) / span;
        (bucket as i32).clamp(0, self.n_buckets as i32 - 1) as usize
    }
}

pub struct FilmTile {
    pub pixel_bounds: Bounds2i,
    pub filter_radius: Vector2f,
    pub inv_filter_radius: Vector2f,
    pub filter_table: Arc<[Float; FT_SZ]>,
    pub pixels: Vec<FilmTilePixel>,
    pub max_sample_luminance: Float,
    pub pixel_sensor: PixelSensor,
    /// Spectral accumulation activated by SpectralFilm; otherwise `None`.
    /// The `spectral_buckets` / `spectral_bucket_weights` buffers are sized
    /// `width * height * n_buckets` when active, empty otherwise.
    pub spectral: Option<SpectralTileConfig>,
    pub spectral_buckets: Vec<Float>,
    pub spectral_bucket_weights: Vec<Float>,
}

impl FilmTile {
    pub fn new(
        pixel_bounds: &Bounds2i,
        filter_radius: &Vector2f,
        filter_table: &Arc<[Float; FT_SZ]>,
        max_sample_luminance: Float,
        pixel_sensor: PixelSensor,
    ) -> Self {
        Self::with_spectral(
            pixel_bounds,
            filter_radius,
            filter_table,
            max_sample_luminance,
            pixel_sensor,
            None,
        )
    }

    /// Same as `new` but lets `SpectralFilm` plug in a `SpectralTileConfig`
    /// so the tile carries per-bucket wavelength accumulators alongside
    /// the RGB-projected contribution.
    pub fn with_spectral(
        pixel_bounds: &Bounds2i,
        filter_radius: &Vector2f,
        filter_table: &Arc<[Float; FT_SZ]>,
        max_sample_luminance: Float,
        pixel_sensor: PixelSensor,
        spectral: Option<SpectralTileConfig>,
    ) -> Self {
        let inv_filter_radius = Vector2f::new(1.0 / filter_radius.x, 1.0 / filter_radius.y);
        let area = pixel_bounds.area() as usize;
        let (spectral_buckets, spectral_bucket_weights) = match spectral {
            Some(cfg) => (
                vec![0.0; area * cfg.n_buckets],
                vec![0.0; area * cfg.n_buckets],
            ),
            None => (Vec::new(), Vec::new()),
        };
        FilmTile {
            pixel_bounds: *pixel_bounds,
            filter_radius: *filter_radius,
            inv_filter_radius,
            filter_table: Arc::clone(filter_table),
            pixels: vec![FilmTilePixel::zero(); area],
            max_sample_luminance,
            pixel_sensor,
            spectral,
            spectral_buckets,
            spectral_bucket_weights,
        }
    }

    //fn i2f(v: &Vector2i) -> Vector2f {
    //    Vector2f::new(v.x as Float, v.y as Float)

    fn f2i(v: &Vector2f) -> Vector2i {
        Vector2i::new(v.x as i32, v.y as i32)
    }

    fn ceil(v: &Vector2f) -> Vector2i {
        Vector2i::new(v.x.ceil() as i32, v.y.ceil() as i32)
    }

    fn floor(v: &Vector2f) -> Vector2i {
        Vector2i::new(v.x.floor() as i32, v.y.floor() as i32)
    }

    fn min(a: &Vector2i, b: &Vector2i) -> Vector2i {
        return Vector2i::new(i32::min(a.x, b.x), i32::min(a.y, b.y));
    }

    fn max(a: &Vector2i, b: &Vector2i) -> Vector2i {
        return Vector2i::new(i32::max(a.x, b.x), i32::max(a.y, b.y));
    }

    fn sample_rgb_with_wavelengths(
        &self,
        l: &Spectrum,
        lambda: Option<&SampledWavelengths>,
    ) -> [Float; 3] {
        let mut rgb = match lambda {
            Some(lambda) => self.pixel_sensor.to_output_rgb_with_wavelengths(l, lambda),
            None => self.pixel_sensor.to_output_rgb(l),
        };
        let m = rgb[0].max(rgb[1]).max(rgb[2]);
        if m > self.max_sample_luminance {
            let scale = self.max_sample_luminance / m;
            rgb[0] *= scale;
            rgb[1] *= scale;
            rgb[2] *= scale;
        }
        rgb
    }

    fn sample_rgb_from_packet(
        &self,
        l: &SampledSpectrum,
        lambda: &SampledWavelengths,
    ) -> [Float; 3] {
        // pbrt-v4 `RGBFilm::AddSample` (film.h:241-247) clamps in
        // **sensor RGB** (= XYZ for `cie1931`), *then* the read-out
        // path applies `outputRGBFromSensorRGB` (film.h:271). r4 used
        // to clamp in the output sRGB space; that over-clamps any
        // bright sample because the colour-space matrix has negative
        // entries and can boost sensor components into output space
        // (pavilion-night's max R dropped from 38 → 6 due to this).
        // Clamp first in sensor RGB and then apply the matrix per
        // sample (equivalent to v4's "matrix at read-out" since the
        // matrix is linear, but lets r4 keep its
        // `add_weighted_sample_contribution` path that already works
        // in output RGB).
        let mut sensor = self.pixel_sensor.to_sensor_rgb_from_packet(l, lambda);
        let m = sensor[0].max(sensor[1]).max(sensor[2]);
        if m > self.max_sample_luminance {
            let scale = self.max_sample_luminance / m;
            sensor[0] *= scale;
            sensor[1] *= scale;
            sensor[2] *= scale;
        }
        self.pixel_sensor.apply_output_matrix(&sensor)
    }

    /// pbrt-v4 `FilmTile::AddSample(pFilm, L, lambda, visibleSurface,
    /// weight)`. The sample's `SampledSpectrum` is the canonical radiance
    /// representation; we do not build a dense `Spectrum` projection
    /// because every consumer below uses the packet directly.
    pub fn add_sample(
        &mut self,
        p_film: &Point2f,
        l: SampledSpectrum,
        lambda: &SampledWavelengths,
        visible_surface: Option<&VisibleSurface>,
        sample_weight: Float,
    ) {
        self.add_sample_filter(p_film, &l, lambda, sample_weight, visible_surface);
    }

    /// pbrt-v4 `RGBFilm::AddSample(pFilm, L, lambda, visibleSurface,
    /// weight)`: after filter importance sampling, the sample contributes
    /// only to the owning pixel. The filter value is carried by
    /// `sample_weight`.
    pub fn add_sample_pixel(
        &mut self,
        p_pixel: &Point2i,
        l: SampledSpectrum,
        lambda: &SampledWavelengths,
        visible_surface: Option<&VisibleSurface>,
        sample_weight: Float,
    ) {
        if p_pixel.x < self.pixel_bounds.min.x
            || self.pixel_bounds.max.x <= p_pixel.x
            || p_pixel.y < self.pixel_bounds.min.y
            || self.pixel_bounds.max.y <= p_pixel.y
        {
            return;
        }

        let sample_rgb = self.sample_rgb_from_packet(&l, lambda);
        let width = (self.pixel_bounds.max.x - self.pixel_bounds.min.x) as usize;
        let xx = (p_pixel.x - self.pixel_bounds.min.x) as usize;
        let yy = (p_pixel.y - self.pixel_bounds.min.y) as usize;
        let pindex = yy * width + xx;

        self.pixels[pindex].contrib_sum[0] += sample_rgb[0] * sample_weight;
        self.pixels[pindex].contrib_sum[1] += sample_rgb[1] * sample_weight;
        self.pixels[pindex].contrib_sum[2] += sample_rgb[2] * sample_weight;
        self.pixels[pindex].filter_weight_sum += sample_weight;

        if let Some(vs) = visible_surface {
            if vs.set {
                let pixel = &mut self.pixels[pindex];
                let p = [vs.p.x, vs.p.y, vs.p.z];
                let n = [vs.n.x, vs.n.y, vs.n.z];
                let ns = [vs.ns.x, vs.ns.y, vs.ns.z];
                let albedo = gbuffer_albedo_to_rgb(vs.albedo, lambda);
                for i in 0..3 {
                    pixel.p_sum[i] += sample_weight * p[i];
                    pixel.n_sum[i] += sample_weight * n[i];
                    pixel.ns_sum[i] += sample_weight * ns[i];
                    pixel.albedo_sum[i] += sample_weight * albedo[i];
                }
                pixel.dzdx_sum += sample_weight * vs.dpdx.z;
                pixel.dzdy_sum += sample_weight * vs.dpdy.z;
                pixel.uv_sum[0] += sample_weight * vs.uv.x;
                pixel.uv_sum[1] += sample_weight * vs.uv.y;
                pixel.gbuffer_weight_sum += sample_weight;
            }
        }

        if let Some(cfg) = self.spectral {
            let n_buckets = cfg.n_buckets;
            for i in 0..N_SPECTRUM_SAMPLES {
                let b = cfg.lambda_to_bucket(lambda[i]);
                let off = pindex * n_buckets + b;
                self.spectral_buckets[off] += l[i] * sample_weight * CIE_Y_INTEGRAL;
                self.spectral_bucket_weights[off] += sample_weight;
            }
        }
    }

    /// Build the per-pixel filter weights for the footprint `[p0, p1)`
    /// centered on `p_film_discrete`. Returns `(ifx, ify, weights)` with
    /// normalized weights. Returns `None` if every weight ended up zero
    /// (sample lands outside the filter support).
    ///
    /// Owning the `filter_table` read lock inside this helper means the
    /// lock is released by the time the caller starts mutating
    /// `self.pixels`, avoiding the borrow conflict that would otherwise
    /// force an explicit `drop`.
    fn compute_footprint_weights(
        &self,
        p_film_discrete: Point2f,
        p0: Vector2i,
        p1: Vector2i,
    ) -> Option<(Vec<i32>, Vec<i32>, Vec<Float>)> {
        let filter_radius = self.filter_radius;
        let inv_filter_radius = self.inv_filter_radius;
        let filter_table_size = FT_W;
        let delta = p1 - p0;

        let mut ifx: Vec<i32> = vec![0; delta.x as usize];
        let lx = inv_filter_radius.x * (filter_table_size - 1) as Float;
        for x in p0.x..p1.x {
            let d = Float::abs(x as Float + 0.5 - p_film_discrete.x);
            let id = if d <= filter_radius.x {
                i32::min(Float::floor(d * lx) as i32, (filter_table_size - 1) as i32)
            } else {
                -1
            };
            ifx[(x - p0.x) as usize] = id;
        }

        let mut ify: Vec<i32> = vec![0; delta.y as usize];
        let ly = inv_filter_radius.y * (filter_table_size - 1) as Float;
        for y in p0.y..p1.y {
            let d = Float::abs(y as Float + 0.5 - p_film_discrete.y);
            let id = if d <= filter_radius.y {
                i32::min(Float::floor(d * ly) as i32, (filter_table_size - 1) as i32)
            } else {
                -1
            };
            ify[(y - p0.y) as usize] = id;
        }

        let mut weights = vec![0.0; ifx.len() * ify.len()];
        {
            let filter_table = &*self.filter_table;
            for y in p0.y..p1.y {
                let yidx = (y - p0.y) as usize;
                let iy = ify[yidx];
                if iy < 0 {
                    continue;
                }
                for x in p0.x..p1.x {
                    let xidx = (x - p0.x) as usize;
                    let ix = ifx[xidx];
                    if ix < 0 {
                        continue;
                    }
                    let offset = (iy * filter_table_size as i32 + ix) as usize;
                    weights[yidx * ifx.len() + xidx] = filter_table[offset];
                }
            }
        }
        let sum = weights.iter().sum::<Float>();
        if sum <= 0.0 {
            return None;
        }
        let isum = 1.0 / sum;
        for w in weights.iter_mut() {
            *w *= isum;
        }
        Some((ifx, ify, weights))
    }

    fn accumulate_spectral_buckets(
        &mut self,
        p0: Vector2i,
        p1: Vector2i,
        ifx_len: usize,
        weights: &[Float],
        l_packet: &SampledSpectrum,
        lambda: &SampledWavelengths,
        sample_weight: Float,
        cfg: SpectralTileConfig,
    ) {
        let width = (self.pixel_bounds.max.x - self.pixel_bounds.min.x) as usize;
        let n_buckets = cfg.n_buckets;
        for y in p0.y..p1.y {
            for x in p0.x..p1.x {
                let xidx = (x - p0.x) as usize;
                let yidx = (y - p0.y) as usize;
                let filter_weight = weights[yidx * ifx_len + xidx];
                if filter_weight == 0.0 {
                    continue;
                }
                let xx = (x - self.pixel_bounds.min.x) as usize;
                let yy = (y - self.pixel_bounds.min.y) as usize;
                let pindex: usize = yy * width + xx;
                let weight_contrib = sample_weight * filter_weight;
                for i in 0..N_SPECTRUM_SAMPLES {
                    let lambda_i = lambda[i];
                    let l_i = l_packet[i];
                    let b = cfg.lambda_to_bucket(lambda_i);
                    let off = pindex * n_buckets + b;
                    self.spectral_buckets[off] += l_i * weight_contrib * CIE_Y_INTEGRAL;
                    self.spectral_bucket_weights[off] += weight_contrib;
                }
            }
        }
    }

    fn add_weighted_sample_contribution(
        &mut self,
        p0: Vector2i,
        p1: Vector2i,
        ifx: &[i32],
        weights: &[Float],
        sample_rgb: &[Float; 3],
        sample_weight: Float,
        visible_surface: Option<&VisibleSurface>,
        lambda: &SampledWavelengths,
    ) {
        let width = (self.pixel_bounds.max.x - self.pixel_bounds.min.x) as usize;
        for y in p0.y..p1.y {
            for x in p0.x..p1.x {
                let xidx = (x - p0.x) as usize;
                let yidx = (y - p0.y) as usize;
                let filter_weight = weights[yidx * ifx.len() + xidx];

                let xx = x - self.pixel_bounds.min.x;
                let yy = y - self.pixel_bounds.min.y;

                assert!(xx >= 0);
                assert!(yy >= 0);

                let xx = xx as usize;
                let yy = yy as usize;

                let pindex: usize = yy * width + xx;
                self.pixels[pindex].contrib_sum[0] += sample_rgb[0] * sample_weight * filter_weight;
                self.pixels[pindex].contrib_sum[1] += sample_rgb[1] * sample_weight * filter_weight;
                self.pixels[pindex].contrib_sum[2] += sample_rgb[2] * sample_weight * filter_weight;
                self.pixels[pindex].filter_weight_sum += filter_weight;
                if let Some(vs) = visible_surface {
                    if vs.set {
                        let pixel = &mut self.pixels[pindex];
                        let p = [vs.p.x, vs.p.y, vs.p.z];
                        let n = [vs.n.x, vs.n.y, vs.n.z];
                        let ns = [vs.ns.x, vs.ns.y, vs.ns.z];
                        let albedo = gbuffer_albedo_to_rgb(vs.albedo, lambda);
                        for i in 0..3 {
                            pixel.p_sum[i] += filter_weight * p[i];
                            pixel.n_sum[i] += filter_weight * n[i];
                            pixel.ns_sum[i] += filter_weight * ns[i];
                            pixel.albedo_sum[i] += filter_weight * albedo[i];
                        }
                        pixel.dzdx_sum += filter_weight * vs.dpdx.z;
                        pixel.dzdy_sum += filter_weight * vs.dpdy.z;
                        pixel.uv_sum[0] += filter_weight * vs.uv.x;
                        pixel.uv_sum[1] += filter_weight * vs.uv.y;
                        pixel.gbuffer_weight_sum += filter_weight;
                    }
                }
            }
        }
    }

    fn add_sample_filter(
        &mut self,
        p_film: &Point2f,
        l: &SampledSpectrum,
        lambda: &SampledWavelengths,
        sample_weight: Float,
        visible_surface: Option<&VisibleSurface>,
    ) {
        let sample_rgb = self.sample_rgb_from_packet(l, lambda);
        let p_film_discrete = *p_film;

        let p0 = Self::floor(&(p_film_discrete - self.filter_radius));
        let p1 = Self::ceil(&(p_film_discrete + self.filter_radius));

        assert!(p1.x - p0.x > 0);
        assert!(p1.y - p0.y > 0);

        let p0 = Self::max(&p0, &self.pixel_bounds.min);
        let p1 = Self::min(&p1, &self.pixel_bounds.max);

        let delta = p1 - p0;
        if delta.x <= 0 || delta.y <= 0 {
            return;
        }

        let Some((ifx, _ify, weights)) = self.compute_footprint_weights(p_film_discrete, p0, p1)
        else {
            return;
        };

        self.add_weighted_sample_contribution(
            p0,
            p1,
            &ifx,
            &weights,
            &sample_rgb,
            sample_weight,
            visible_surface,
            lambda,
        );

        if let Some(cfg) = self.spectral {
            self.accumulate_spectral_buckets(
                p0,
                p1,
                ifx.len(),
                &weights,
                l,
                lambda,
                sample_weight,
                cfg,
            );
        }
    }

    pub fn add_sample_single(&mut self, p_film: &Point2f, l: &Spectrum, _sample_weight: Float) {
        let sample_rgb = self.sample_rgb_with_wavelengths(l, None);

        let p = Self::f2i(p_film);
        let x = p.x;
        let y = p.y;

        if x < self.pixel_bounds.min.x || self.pixel_bounds.max.x <= x {
            return;
        }
        if y < self.pixel_bounds.min.y || self.pixel_bounds.max.y <= y {
            return;
        }
        let width = (self.pixel_bounds.max.x - self.pixel_bounds.min.x) as usize;

        let xx = (x - self.pixel_bounds.min.x) as usize;
        let yy = (y - self.pixel_bounds.min.y) as usize;

        let pindex = yy * width + xx;
        self.pixels[pindex].contrib_sum[0] += sample_rgb[0];
        self.pixels[pindex].contrib_sum[1] += sample_rgb[1];
        self.pixels[pindex].contrib_sum[2] += sample_rgb[2];
        self.pixels[pindex].filter_weight_sum += 1.0;
    }

    pub fn get_pixel_index(&self, p: &Vector2i) -> usize {
        let width = self.pixel_bounds.max.x - self.pixel_bounds.min.x;
        let x = p.x - self.pixel_bounds.min.x;
        let y = p.y - self.pixel_bounds.min.y;
        return (y * width + x) as usize;
    }

    pub fn get_pixel_bounds(&self) -> Bounds2i {
        return self.pixel_bounds;
    }
}
