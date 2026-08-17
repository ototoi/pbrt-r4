use super::mipmap_weight_lut::MIPMAP_WEIGHT_LUT;
use super::noise::*;
use crate::base::texture::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::imageio::*;
use crate::util::profile::*;
use crate::util::spectrum::*;
use crate::util::stats::*;
use std::mem::size_of;

use log::*;
use rayon::prelude::*;
use std::fmt::Debug;
use std::vec;

thread_local!(static N_EWA_LOOKUPS: StatCounter = StatCounter::new("Texture/EWA lookups"));
thread_local!(static N_TRILERP_LOOKUPS: StatCounter = StatCounter::new("Texture/Trilinear lookups"));
thread_local!(static MIP_MAP_MEMORY: StatMemoryCounter = StatMemoryCounter::new("Memory/Texture MIP maps"));

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MIPMapStorageKind {
    F32,
    F16,
    U8 { encoding: ColorEncoding },
}

impl MIPMapStorageKind {
    pub fn from_raw_image(raw: &RawImage) -> Self {
        match &raw.data {
            RawImageData::F32(_) => Self::F32,
            RawImageData::F16(_) => Self::F16,
            RawImageData::U8 { encoding, .. } => Self::U8 {
                encoding: *encoding,
            },
        }
    }
}

pub enum MIPMapLevelStorage {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    U8 {
        data: Vec<u8>,
        encoding: ColorEncoding,
    },
}

impl MIPMapLevelStorage {
    fn from_f32(data: Vec<f32>, kind: MIPMapStorageKind) -> Self {
        match kind {
            MIPMapStorageKind::F32 => MIPMapLevelStorage::F32(data),
            MIPMapStorageKind::F16 => {
                MIPMapLevelStorage::F16(data.into_iter().map(half::f16::from_f32).collect())
            }
            MIPMapStorageKind::U8 { encoding } => MIPMapLevelStorage::U8 {
                data: data
                    .into_iter()
                    .map(|v| {
                        let v = encoding.from_linear(v as Float);
                        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
                    })
                    .collect(),
                encoding,
            },
        }
    }

    fn lookup(&self, i: usize) -> f32 {
        match self {
            MIPMapLevelStorage::F32(data) => data[i],
            MIPMapLevelStorage::F16(data) => data[i].to_f32(),
            MIPMapLevelStorage::U8 { data, encoding } => {
                let v = data[i] as Float / 255.0;
                encoding.to_linear(v) as f32
            }
        }
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            MIPMapLevelStorage::F32(data) => data.clone(),
            MIPMapLevelStorage::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
            MIPMapLevelStorage::U8 { data, encoding } => data
                .iter()
                .map(|v| {
                    let v = *v as Float / 255.0;
                    encoding.to_linear(v) as f32
                })
                .collect(),
        }
    }

    fn bytes_used(&self) -> usize {
        match self {
            MIPMapLevelStorage::F32(data) => data.len() * size_of::<f32>(),
            MIPMapLevelStorage::F16(data) => data.len() * size_of::<half::f16>(),
            MIPMapLevelStorage::U8 { data, .. } => data.len(),
        }
    }
}

pub struct F32MIPMapImage {
    pub resolution: (usize, usize),
    pub channels: usize,
    pub data: MIPMapLevelStorage,
}

impl F32MIPMapImage {
    pub fn new(data: Vec<f32>, resolution: (usize, usize)) -> Self {
        let channels = data.len() / (resolution.0 * resolution.1);
        F32MIPMapImage {
            resolution,
            channels,
            data: MIPMapLevelStorage::F32(data),
        }
    }

    fn from_f32_storage(
        data: Vec<f32>,
        resolution: (usize, usize),
        channels: usize,
        storage: MIPMapStorageKind,
    ) -> Self {
        F32MIPMapImage {
            resolution,
            channels,
            data: MIPMapLevelStorage::from_f32(data, storage),
        }
    }
}

impl From<(&[f32], (usize, usize))> for F32MIPMapImage {
    fn from(v: (&[f32], (usize, usize))) -> Self {
        F32MIPMapImage {
            resolution: v.1,
            channels: v.0.len() / (v.1 .0 * v.1 .1),
            data: MIPMapLevelStorage::F32(v.0.to_vec()),
        }
    }
}

impl From<(&[f64], (usize, usize))> for F32MIPMapImage {
    fn from(v: (&[f64], (usize, usize))) -> Self {
        let data: Vec<f32> = v.0.iter().map(|x| *x as f32).collect();
        F32MIPMapImage {
            resolution: v.1,
            channels: data.len() / (v.1 .0 * v.1 .1),
            data: MIPMapLevelStorage::F32(data),
        }
    }
}

impl From<(&[RGBSpectrum], (usize, usize))> for F32MIPMapImage {
    fn from(v: (&[RGBSpectrum], (usize, usize))) -> Self {
        let mut data = vec![0.0; 3 * v.0.len()];
        for i in 0..v.0.len() {
            let c = v.0[i].to_rgb();
            data[3 * i + 0] = c[0] as f32;
            data[3 * i + 1] = c[1] as f32;
            data[3 * i + 2] = c[2] as f32;
        }
        F32MIPMapImage {
            resolution: v.1,
            channels: 3,
            data: MIPMapLevelStorage::F32(data),
        }
    }
}

pub trait MIPMapImage<T> {
    fn lookup(&self, i: usize) -> T;
    fn bilerp(&self, st: &Point2f, swrap_mode: ImageWrap, twrap_mode: ImageWrap) -> T;
    fn get_width(&self) -> usize;
    fn get_height(&self) -> usize;
    fn as_data(&self) -> &F32MIPMapImage;
}

impl MIPMapImage<Float> for F32MIPMapImage {
    fn lookup(&self, i: usize) -> Float {
        return self.data.lookup(self.channels * i) as Float;
    }
    fn bilerp(&self, st: &Point2f, swrap_mode: ImageWrap, twrap_mode: ImageWrap) -> Float {
        match self.channels {
            1 => bilerp_channel(self, st, 0, swrap_mode, twrap_mode),
            3 => {
                let r = bilerp_channel(self, st, 0, swrap_mode, twrap_mode);
                let g = bilerp_channel(self, st, 1, swrap_mode, twrap_mode);
                let b = bilerp_channel(self, st, 2, swrap_mode, twrap_mode);
                (r + g + b) / 3.0
            }
            4 => bilerp_channel(self, st, 3, swrap_mode, twrap_mode),
            channels => panic!("unexpected Float MIPMap channel count: {channels}"),
        }
    }
    fn get_width(&self) -> usize {
        self.resolution.0
    }
    fn get_height(&self) -> usize {
        self.resolution.1
    }
    fn as_data(&self) -> &F32MIPMapImage {
        return self;
    }
}

impl MIPMapImage<RGBSpectrum> for F32MIPMapImage {
    fn lookup(&self, i: usize) -> RGBSpectrum {
        match self.channels {
            1 => {
                let v = self.data.lookup(i) as Float;
                RGBSpectrum::rgb_from_rgb(&[v, v, v])
            }
            3 | 4 => RGBSpectrum::rgb_from_rgb(&[
                self.data.lookup(self.channels * i) as Float,
                self.data.lookup(self.channels * i + 1) as Float,
                self.data.lookup(self.channels * i + 2) as Float,
            ]),
            channels => panic!("unexpected RGB MIPMap channel count: {channels}"),
        }
    }
    fn bilerp(&self, st: &Point2f, swrap_mode: ImageWrap, twrap_mode: ImageWrap) -> RGBSpectrum {
        match self.channels {
            1 => {
                let v = bilerp_channel(self, st, 0, swrap_mode, twrap_mode);
                RGBSpectrum::rgb_from_rgb(&[v, v, v])
            }
            3 | 4 => RGBSpectrum::rgb_from_rgb(&[
                bilerp_channel(self, st, 0, swrap_mode, twrap_mode),
                bilerp_channel(self, st, 1, swrap_mode, twrap_mode),
                bilerp_channel(self, st, 2, swrap_mode, twrap_mode),
            ]),
            channels => panic!("unexpected RGB MIPMap channel count: {channels}"),
        }
    }
    fn get_width(&self) -> usize {
        self.resolution.0
    }
    fn get_height(&self) -> usize {
        self.resolution.1
    }
    fn as_data(&self) -> &F32MIPMapImage {
        return self;
    }
}

struct ResampleWeight {
    first_texel: i32,
    weight: [f32; 4],
}

fn math_mod(a: i32, b: i32) -> i32 {
    let result = a - (a / b) * b;
    return if result < 0 { result + b } else { result };
}

fn remap_octahedral(mut s: i32, mut t: i32, w: i32, h: i32) -> (i32, i32) {
    if s < 0 {
        s = -s;
        t = h - 1 - t;
    } else if s >= w {
        s = 2 * w - 1 - s;
        t = h - 1 - t;
    }
    if t < 0 {
        s = w - 1 - s;
        t = -t;
    } else if t >= h {
        s = w - 1 - s;
        t = 2 * h - 1 - t;
    }
    if w == 1 {
        s = 0;
    }
    if h == 1 {
        t = 0;
    }
    (s, t)
}

fn channel_texel_static(
    image: &F32MIPMapImage,
    s: i32,
    t: i32,
    channel: usize,
    swrap_mode: ImageWrap,
    twrap_mode: ImageWrap,
) -> Float {
    let w = image.resolution.0 as i32;
    let h = image.resolution.1 as i32;
    let (mut s, mut t) = (s, t);
    if swrap_mode == ImageWrap::OctahedralSphere || twrap_mode == ImageWrap::OctahedralSphere {
        (s, t) = remap_octahedral(s, t, w, h);
    } else {
        match swrap_mode {
            ImageWrap::Repeat => s &= w - 1,
            ImageWrap::Clamp => s = i32::clamp(s, 0, w - 1),
            _ if s < 0 || w <= s => return 0.0,
            _ => {}
        }
        match twrap_mode {
            ImageWrap::Repeat => t &= h - 1,
            ImageWrap::Clamp => t = i32::clamp(t, 0, h - 1),
            _ if t < 0 || h <= t => return 0.0,
            _ => {}
        }
    }
    let index = (t * w + s) as usize;
    image.data.lookup(image.channels * index + channel) as Float
}

fn bilerp_channel(
    image: &F32MIPMapImage,
    st: &Point2f,
    channel: usize,
    swrap_mode: ImageWrap,
    twrap_mode: ImageWrap,
) -> Float {
    let s = st[0] * image.resolution.0 as Float - 0.5;
    let t = st[1] * image.resolution.1 as Float - 0.5;
    let s0 = s.floor() as i32;
    let t0 = t.floor() as i32;
    let ds = s - s0 as Float;
    let dt = t - t0 as Float;
    let a = channel_texel_static(image, s0, t0, channel, swrap_mode, twrap_mode);
    let b = channel_texel_static(image, s0, t0 + 1, channel, swrap_mode, twrap_mode);
    let c = channel_texel_static(image, s0 + 1, t0, channel, swrap_mode, twrap_mode);
    let d = channel_texel_static(image, s0 + 1, t0 + 1, channel, swrap_mode, twrap_mode);
    a * ((1.0 - ds) * (1.0 - dt)) + b * ((1.0 - ds) * dt) + c * (ds * (1.0 - dt)) + d * (ds * dt)
}

/// Write a 2x-downsampled (along x) copy of `src` into `dst`. `dst` is
/// resized in place; if its capacity already covers the new size, no
/// allocation happens. This lets `make_pyramid` reuse the same two
/// ping-pong buffers across every level of the pyramid instead of
/// allocating a fresh `Vec<f32>` per level (which on a 5760x2880 HDR
/// sky map was responsible for ~4 GB of peak RSS during scene build).
fn downsample_half_into(
    src: &[f32],
    channels: usize,
    width: usize,
    height: usize,
    dst: &mut Vec<f32>,
) {
    let nw = width / 2;
    let nh = height;
    let new_size = channels * nw * nh;
    dst.clear();
    dst.resize(new_size, 0.0);
    // Parallel writers into non-overlapping row slices — no per-row
    // `Vec::new()` allocations like the previous `.par_iter().collect()`
    // implementation.
    dst.par_chunks_mut(channels * nw)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..nw {
                for c in 0..channels {
                    let i0 = y * width + (2 * x + 0);
                    let i1 = y * width + (2 * x + 1);
                    row[channels * x + c] =
                        src[channels * i0 + c] * 0.5 + src[channels * i1 + c] * 0.5;
                }
            }
        });
}

/// Write the transpose of `src` (of `(width, height)` shape with
/// `channels` interleaved channels) into `dst`. `dst` is resized in
/// place.
fn transpose_image_into(
    src: &[f32],
    channels: usize,
    width: usize,
    height: usize,
    dst: &mut Vec<f32>,
) {
    let size = channels * width * height;
    dst.clear();
    dst.resize(size, 0.0);
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let s = channels * (y * width + x) + c;
                let d = channels * (x * height + y) + c;
                dst[d] = src[s];
            }
        }
    }
}

fn resample_weights(old_res: usize, new_res: usize) -> Vec<ResampleWeight> {
    let mut wt = Vec::with_capacity(new_res);
    const FILTER_WIDTH: Float = 2.0;
    for i in 0..new_res {
        let center: Float = (i as Float + 0.5) * (old_res as Float / new_res as Float);
        let first_texel = Float::floor((center - FILTER_WIDTH) + 0.5);
        let mut weight = [0.0; 4];
        for j in 0..4 {
            let pos = first_texel + (j as Float) + 0.5;
            weight[j] = lanczos((pos - center) / FILTER_WIDTH, FILTER_WIDTH) as f32;
        }
        // Normalize filter weights for texel resampling
        let inv_sum_wts = 1.0 / (weight[0] + weight[1] + weight[2] + weight[3]);
        for j in 0..4 {
            weight[j] *= inv_sum_wts;
        }
        let resampled = ResampleWeight {
            first_texel: (first_texel as i32),
            weight,
        };
        wt.push(resampled);
    }
    return wt;
}

/// Resample `img` to the next power-of-two resolution. Takes the
/// input by value so the pow-of-2 fast path is a zero-copy return,
/// and the non-pow-2 path can drop the original allocation as soon as
/// its data has been copied into the resampled buffer (instead of
/// holding both for the whole resample loop, which on sky.exr-class
/// inputs added ~200 MB to peak RSS for nothing).
fn resample_image(
    img: Vec<f32>,
    channels: usize,
    resolution: (usize, usize),
    swrap_mode: ImageWrap,
    twrap_mode: ImageWrap,
) -> ((usize, usize), Vec<f32>) {
    if is_power_of_2(resolution.0 as u32) && is_power_of_2(resolution.1 as u32) {
        return (resolution, img);
    }
    let res_pow2 = (
        round_up_pow2(resolution.0 as u32) as usize,
        round_up_pow2(resolution.1 as u32) as usize,
    );
    if swrap_mode == ImageWrap::OctahedralSphere && twrap_mode == ImageWrap::OctahedralSphere {
        return resample_image_octahedral(img, channels, resolution, res_pow2);
    }
    let ratio = (res_pow2.0 * res_pow2.1) as Float / (resolution.0 * resolution.1) as Float;
    info!(
        "Resampling MIPMap from {:?} to {:?}. Ratio= {}",
        resolution, res_pow2, ratio
    );
    let mut resampled_image = vec![0.0; channels * res_pow2.0 * res_pow2.1];
    {
        let s_weights = resample_weights(resolution.0, res_pow2.0);
        for t in 0..resolution.1 {
            for s in 0..res_pow2.0 {
                for j in 0..4 {
                    let mut orig_s = s_weights[s].first_texel + j;
                    if swrap_mode == ImageWrap::Repeat {
                        orig_s = math_mod(orig_s as i32, resolution.0 as i32);
                    } else if swrap_mode == ImageWrap::Clamp {
                        orig_s = i32::clamp(orig_s as i32, 0, resolution.0 as i32 - 1);
                    }
                    if orig_s >= 0 && orig_s < resolution.0 as i32 {
                        let w = s_weights[s].weight[j as usize];
                        let src = t * resolution.0 + orig_s as usize;
                        let dst = (t * res_pow2.0 + s) as usize;
                        for c in 0..channels {
                            resampled_image[channels * dst + c] += img[channels * src + c] * w;
                        }
                    }
                }
            }
        }
    }
    // Free the input buffer as soon as the x-axis resample is done —
    // the y-axis resample below works in-place on `resampled_image`.
    drop(img);

    {
        let t_weights = resample_weights(resolution.1, res_pow2.1);
        // Single column scratch reused for every column instead of
        // re-allocating inside the loop.
        let mut buffer: Vec<f32> = vec![0.0; channels * resolution.1];
        for s in 0..res_pow2.0 {
            for t in 0..resolution.1 {
                let src = (t * res_pow2.0 + s) as usize;
                let dst = t;
                for c in 0..channels {
                    buffer[channels * dst + c] = resampled_image[channels * src + c];
                }
            }

            for t in 0..res_pow2.1 {
                let mut l = vec![0.0; channels];
                for j in 0..4 {
                    let mut orig_t = t_weights[t].first_texel + j;
                    if twrap_mode == ImageWrap::Repeat {
                        orig_t = math_mod(orig_t as i32, resolution.1 as i32);
                    } else if twrap_mode == ImageWrap::Clamp {
                        orig_t = i32::clamp(orig_t as i32, 0, resolution.1 as i32 - 1);
                    }

                    if orig_t >= 0 && orig_t < resolution.1 as i32 {
                        let w = t_weights[t as usize].weight[j as usize];
                        let src = orig_t as usize;
                        for c in 0..channels {
                            l[c] += buffer[channels * src + c] * w;
                        }
                    }
                }
                let dst = (t * res_pow2.0 + s) as usize;
                for c in 0..channels {
                    resampled_image[channels * dst + c] = l[c];
                }
            }
        }
    }
    for v in resampled_image.iter_mut() {
        *v = v.max(0.0);
    }
    (res_pow2, resampled_image)
}

fn resample_image_octahedral(
    img: Vec<f32>,
    channels: usize,
    resolution: (usize, usize),
    res_pow2: (usize, usize),
) -> ((usize, usize), Vec<f32>) {
    let x_weights = resample_weights(resolution.0, res_pow2.0);
    let y_weights = resample_weights(resolution.1, res_pow2.1);
    let mut result = vec![0.0; channels * res_pow2.0 * res_pow2.1];
    for y in 0..res_pow2.1 {
        for x in 0..res_pow2.0 {
            for jy in 0..4 {
                for jx in 0..4 {
                    let (sx, sy) = remap_octahedral(
                        x_weights[x].first_texel + jx,
                        y_weights[y].first_texel + jy,
                        resolution.0 as i32,
                        resolution.1 as i32,
                    );
                    if sx < 0 || sx >= resolution.0 as i32 || sy < 0 || sy >= resolution.1 as i32 {
                        continue;
                    }
                    let weight =
                        x_weights[x].weight[jx as usize] * y_weights[y].weight[jy as usize];
                    let source = (sy as usize * resolution.0 + sx as usize) * channels;
                    let destination = (y * res_pow2.0 + x) * channels;
                    for c in 0..channels {
                        result[destination + c] += img[source + c] * weight;
                    }
                }
            }
        }
    }
    for value in &mut result {
        *value = value.max(0.0);
    }
    (res_pow2, result)
}

fn make_pyramid<T>(
    pyramid: &mut Vec<F32MIPMapImage>,
    channels: usize,
    resolution: (usize, usize),
    data: Vec<f32>,
    storage: MIPMapStorageKind,
) where
    F32MIPMapImage: MIPMapImage<T>,
{
    // Two ping-pong scratch buffers shared across every downsample /
    // transpose step in the pyramid. Their backing allocations grow up
    // to the size of the largest intermediate (one full base-level
    // image) and then get reused for every subsequent level.
    let mut scratch_a: Vec<f32> = Vec::new();
    let mut scratch_b: Vec<f32> = Vec::new();
    make_pyramid_with_scratch::<T>(
        pyramid,
        channels,
        resolution,
        data,
        storage,
        &mut scratch_a,
        &mut scratch_b,
    );
}

fn make_pyramid_with_scratch<T>(
    pyramid: &mut Vec<F32MIPMapImage>,
    channels: usize,
    resolution: (usize, usize),
    data: Vec<f32>,
    storage: MIPMapStorageKind,
    scratch_a: &mut Vec<f32>,
    scratch_b: &mut Vec<f32>,
) where
    F32MIPMapImage: MIPMapImage<T>,
{
    let total = resolution.0 * resolution.1;
    if total == 1 {
        let image = F32MIPMapImage::from_f32_storage(data, resolution, channels, storage);
        pyramid.push(image);
        return;
    }

    let c = channels;
    let w = resolution.0;
    let h = resolution.1;

    // Two-step downsample (x then y via transpose) using `scratch_a`
    // and `scratch_b` as ping-pong storage. After the dance below,
    // `scratch_a` holds the next pyramid level's image; we then move
    // it into a fresh `Vec` (sized exactly to fit) for permanent
    // storage in the pyramid.
    let (nw, nh) = if w > 1 {
        downsample_half_into(&data, c, w, h, scratch_a);
        (w / 2, h)
    } else {
        scratch_a.clear();
        scratch_a.extend_from_slice(&data);
        (w, h)
    };

    let (nw, nh) = if h > 1 {
        transpose_image_into(scratch_a, c, nw, nh, scratch_b);
        downsample_half_into(scratch_b, c, nh, nw, scratch_a);
        transpose_image_into(scratch_a, c, nh / 2, nw, scratch_b);
        std::mem::swap(scratch_a, scratch_b);
        (nw, nh / 2)
    } else {
        (nw, nh)
    };

    // Push the current level's image (takes ownership of `data`).
    let image = F32MIPMapImage::from_f32_storage(data, resolution, channels, storage);
    pyramid.push(image);

    // Move the contents of `scratch_a` into a tight-fitting `Vec` so
    // the pyramid level owns just what it needs. `std::mem::take`
    // leaves the scratch with its old capacity, ready to be reused
    // (after the recursive call below resizes it as needed).
    let next_data = std::mem::take(scratch_a);
    make_pyramid_with_scratch::<T>(
        pyramid,
        c,
        (nw, nh),
        next_data,
        storage,
        scratch_a,
        scratch_b,
    );
}

fn log2int_(x: usize) -> usize {
    return f32::ceil(f32::log(x as f32, 2.0)) as usize;
}

fn make_mipimages<T>(
    data: &[T],
    resolution: (usize, usize),
    swrap_mode: ImageWrap,
    twrap_mode: ImageWrap,
    storage: MIPMapStorageKind,
) -> Vec<F32MIPMapImage>
where
    T: Clone + Debug,
    F32MIPMapImage: MIPMapImage<T>,
    F32MIPMapImage: for<'a> From<(&'a [T], (usize, usize))>,
{
    // Convert the input from `&[T]` into an owned `Vec<f32>` once,
    // then thread it by value all the way through `resample_image` and
    // `make_pyramid`. The previous code held both `mipdata.data` and a
    // freshly-allocated `resampled_image` simultaneously across the
    // resample loop; on an sky.exr-class 5760×2880 RGB input that's
    // ~400 MB just for the staging phase.
    let mipdata = F32MIPMapImage::from((data, resolution));
    let channels = mipdata.channels;

    make_mipimages_from_f32::<T>(
        mipdata.data.to_f32_vec(),
        channels,
        resolution,
        swrap_mode,
        twrap_mode,
        storage,
    )
}

fn make_mipimages_from_f32<T>(
    data: Vec<f32>,
    channels: usize,
    resolution: (usize, usize),
    swrap_mode: ImageWrap,
    twrap_mode: ImageWrap,
    storage: MIPMapStorageKind,
) -> Vec<F32MIPMapImage>
where
    F32MIPMapImage: MIPMapImage<T>,
{
    let (resolution, data) = resample_image(data, channels, resolution, swrap_mode, twrap_mode);
    // Initialize levels of MIPMap from image
    let n_levels = 1 + log2int_(usize::max(resolution.0, resolution.1));
    let mut pyramid = Vec::with_capacity(n_levels);

    make_pyramid(&mut pyramid, channels, resolution, data, storage);

    pyramid
}

/// Channel-preserving MIP levels and the sampling configuration shared by
/// typed views.  The levels retain the source channel count; conversion to a
/// Float or RGBSpectrum happens only in the view's evaluation methods.
pub struct MIPMapStorage {
    pub pyramid: Vec<F32MIPMapImage>,
}

pub struct MIPMap<T> {
    pub storage: MIPMapStorage,
    pub filter: ImageFilter,
    pub max_anisotropy: Float,
    pub swrap_mode: ImageWrap,
    pub twrap_mode: ImageWrap,
    _marker: std::marker::PhantomData<fn() -> T>,
}

pub type MIPMapFloatView = MIPMap<Float>;
pub type MIPMapSpectrumView = MIPMap<RGBSpectrum>;

impl<
        T: Default + Debug + Copy + std::ops::Add<T, Output = T> + std::ops::Mul<Float, Output = T>,
    > MIPMap<T>
where
    F32MIPMapImage: MIPMapImage<T>,
    F32MIPMapImage: for<'a> From<(&'a [T], (usize, usize))>,
{
    pub fn texel_static(
        image: &F32MIPMapImage,
        s: i32,
        t: i32,
        swrap_mode: ImageWrap,
        twrap_mode: ImageWrap,
    ) -> T {
        let w = <F32MIPMapImage as MIPMapImage<T>>::get_width(image) as i32;
        let h = <F32MIPMapImage as MIPMapImage<T>>::get_height(image) as i32;
        let mut s = s;
        let mut t = t;
        if swrap_mode == ImageWrap::OctahedralSphere || twrap_mode == ImageWrap::OctahedralSphere {
            let (s, t) = remap_octahedral(s, t, w, h);
            let index = (t * w + s) as usize;
            return <F32MIPMapImage as MIPMapImage<T>>::lookup(image, index);
        }
        match swrap_mode {
            ImageWrap::Repeat => {
                s &= w - 1;
            }
            ImageWrap::Clamp => {
                s = i32::clamp(s, 0, w - 1);
            }
            _ => {
                if s < 0 || w <= s {
                    return T::default();
                }
            }
        }
        match twrap_mode {
            ImageWrap::Repeat => {
                t &= h - 1;
            }
            ImageWrap::Clamp => {
                t = i32::clamp(t, 0, h - 1);
            }
            _ => {
                if t < 0 || h <= t {
                    return T::default();
                }
            }
        }
        let index = (t * w + s) as usize;
        return <F32MIPMapImage as MIPMapImage<T>>::lookup(image, index);
    }

    pub fn lerp(t: Float, a: T, b: T) -> T {
        return a * (1.0 - t) + b * t;
    }

    pub fn new(
        resolution: &Point2i,
        data: &[T],
        filter: ImageFilter,
        max_anisotropy: Float,
        swrap_mode: ImageWrap,
        twrap_mode: ImageWrap,
    ) -> Self {
        Self::new_with_storage(
            resolution,
            data,
            filter,
            max_anisotropy,
            swrap_mode,
            twrap_mode,
            MIPMapStorageKind::F32,
        )
    }

    pub fn new_with_storage(
        resolution: &Point2i,
        data: &[T],
        filter: ImageFilter,
        max_anisotropy: Float,
        swrap_mode: ImageWrap,
        twrap_mode: ImageWrap,
        storage: MIPMapStorageKind,
    ) -> Self {
        let _p = ProfilePhase::new(Prof::MIPMapCreation);

        let resolution = (resolution.x as usize, resolution.y as usize);
        let pyramid = make_mipimages(data, resolution, swrap_mode, twrap_mode, storage);
        let mip = Self::make_from_pyramid(pyramid, filter, max_anisotropy, swrap_mode, twrap_mode);
        return mip;
    }

    /// Construct a view over an image whose interleaved channels must be
    /// retained for v4-compatible Float evaluation.  This is intentionally a
    /// separate entry point from `new`, whose typed slice cannot express the
    /// source channel layout.
    pub fn new_with_raw_channels(
        resolution: &Point2i,
        data: &[f32],
        channels: usize,
        filter: ImageFilter,
        max_anisotropy: Float,
        swrap_mode: ImageWrap,
        twrap_mode: ImageWrap,
    ) -> Self {
        Self::new_with_raw_channels_and_storage(
            resolution,
            data,
            channels,
            filter,
            max_anisotropy,
            swrap_mode,
            twrap_mode,
            MIPMapStorageKind::F32,
        )
    }

    pub fn new_from_raw_image(
        raw: &RawImage,
        filter: ImageFilter,
        max_anisotropy: Float,
        swrap_mode: ImageWrap,
        twrap_mode: ImageWrap,
    ) -> Self {
        let data = raw.data_f32();
        Self::new_with_raw_channels_and_storage(
            &raw.resolution,
            &data,
            raw.channels,
            filter,
            max_anisotropy,
            swrap_mode,
            twrap_mode,
            MIPMapStorageKind::from_raw_image(raw),
        )
    }

    pub fn new_with_raw_channels_and_storage(
        resolution: &Point2i,
        data: &[f32],
        channels: usize,
        filter: ImageFilter,
        max_anisotropy: Float,
        swrap_mode: ImageWrap,
        twrap_mode: ImageWrap,
        storage: MIPMapStorageKind,
    ) -> Self {
        assert!((1..=4).contains(&channels));
        let _p = ProfilePhase::new(Prof::MIPMapCreation);
        let resolution = (resolution.x as usize, resolution.y as usize);
        let pyramid = make_mipimages_from_f32::<T>(
            data.to_vec(),
            channels,
            resolution,
            swrap_mode,
            twrap_mode,
            storage,
        );
        Self::make_from_pyramid(pyramid, filter, max_anisotropy, swrap_mode, twrap_mode)
    }

    pub fn make_from_pyramid(
        pyramid: Vec<F32MIPMapImage>,
        filter: ImageFilter,
        max_anisotropy: Float,
        swrap_mode: ImageWrap,
        twrap_mode: ImageWrap,
    ) -> Self {
        {
            let total_consumption: usize =
                pyramid.iter().map(|p| p.as_data().data.bytes_used()).sum();
            MIP_MAP_MEMORY.with(|m| {
                m.add(total_consumption);
            });
        }

        let mip = MIPMap::<T> {
            storage: MIPMapStorage { pyramid },
            filter,
            max_anisotropy,
            swrap_mode,
            twrap_mode,
            _marker: std::marker::PhantomData,
        };
        return mip;
    }

    pub fn width(&self) -> usize {
        return self.storage.pyramid[0].get_width();
    }

    pub fn height(&self) -> usize {
        return self.storage.pyramid[0].get_height();
    }

    pub fn levels(&self) -> usize {
        return self.storage.pyramid.len();
    }

    pub fn texel(&self, level: usize, s: i32, t: i32) -> T {
        let image = &self.storage.pyramid[level];
        return Self::texel_static(image, s, t, self.swrap_mode, self.twrap_mode);
    }

    pub fn lookup(&self, st: &Point2f, width: Float) -> T {
        N_TRILERP_LOOKUPS.with(|n| n.inc());
        let _p = ProfilePhase::new(Prof::TexFiltTrilerp);

        let max_level = (self.levels() - 1) as Float;
        let level = max_level + Float::log2(Float::max(width, 1e-8));
        if level < 0.0 {
            return self.triangle(0, st);
        } else if level >= max_level {
            return self.texel(self.levels() - 1, 0, 0);
        } else {
            let i_level = Float::floor(level) as usize;
            let delta = (level - i_level as Float).clamp(0.0, 1.0);
            let a = self.triangle(i_level, st);
            let b = self.triangle(i_level + 1, st);
            return Self::lerp(delta, a, b);
        }
    }

    pub fn lookup_delta(&self, st: &Point2f, dst0: &Vector2f, dst1: &Vector2f) -> T {
        if self.filter != ImageFilter::EWA {
            let width = Float::max(
                Float::max(Float::abs(dst0[0]), Float::abs(dst0[1])),
                Float::max(Float::abs(dst1[0]), Float::abs(dst1[1])),
            );
            let width = 2.0 * width;
            let max_level = (self.levels() - 1) as Float;
            let level = max_level + Float::log2(Float::max(width, 1e-8));
            if level >= max_level {
                return self.texel(self.levels() - 1, 0, 0);
            }
            let i_level = usize::max(0, level.floor() as usize);
            return match self.filter {
                ImageFilter::Point => {
                    let resolution = &self.storage.pyramid[i_level];
                    let sti = Point2i::new(
                        (st[0] * resolution.get_width() as Float - 0.5).round() as i32,
                        (st[1] * resolution.get_height() as Float - 0.5).round() as i32,
                    );
                    self.texel(i_level, sti.x, sti.y)
                }
                ImageFilter::Bilinear => self.triangle(i_level, st),
                ImageFilter::Trilinear => {
                    if i_level == 0 {
                        self.triangle(0, st)
                    } else {
                        Self::lerp(
                            level - i_level as Float,
                            self.triangle(i_level, st),
                            self.triangle(i_level + 1, st),
                        )
                    }
                }
                ImageFilter::EWA => unreachable!(),
            };
        } else {
            N_EWA_LOOKUPS.with(|n| n.inc());
            let _p = ProfilePhase::new(Prof::TexFiltEWA);
            let mut dst0 = *dst0;
            let mut dst1 = *dst1;
            // Compute ellipse minor and major axes
            if dst0.length_squared() < dst1.length_squared() {
                std::mem::swap(&mut dst0, &mut dst1);
            }
            let major_length = dst0.length(); //longest axis
            let mut minor_length = dst1.length(); //shortest axis

            // Clamp ellipse eccentricity if too large
            if minor_length * self.max_anisotropy < major_length && minor_length > 0.0 {
                let scale = major_length / (minor_length * self.max_anisotropy);
                dst1 *= scale;
                minor_length *= scale;
            }

            if minor_length <= 0.0 {
                return self.triangle(0, st);
            }

            // Choose level of detail for EWA lookup and perform EWA filtering
            let lod = Float::max(
                0.0,
                self.levels() as Float - 1.0 + Float::log2(minor_length),
            );
            let ilod = lod.floor() as usize;
            return Self::lerp(
                lod - ilod as Float,
                self.ewa(ilod, st, dst0, dst1),
                self.ewa(ilod + 1, st, dst0, dst1),
            );
        }
    }

    pub fn ewa(&self, level: usize, st: &Point2f, dst0: Vector2f, dst1: Vector2f) -> T {
        if level >= self.levels() {
            return self.texel(self.levels() - 1, 0, 0);
        }
        // Convert EWA coordinates to appropriate scale for level
        let st = Point2f::new(
            st[0] * self.storage.pyramid[level].get_width() as Float - 0.5,
            st[1] * self.storage.pyramid[level].get_height() as Float - 0.5,
        );
        let dst0 = Vector2f::new(
            dst0[0] * self.storage.pyramid[level].get_width() as Float,
            dst0[1] * self.storage.pyramid[level].get_height() as Float,
        );
        let dst1 = Vector2f::new(
            dst1[0] * self.storage.pyramid[level].get_width() as Float,
            dst1[1] * self.storage.pyramid[level].get_height() as Float,
        );

        // Compute ellipse coefficients to bound EWA filter region
        let a = dst0[1] * dst0[1] + dst1[1] * dst1[1] + 1.0;
        let b = -2.0 * (dst0[0] * dst0[1] + dst1[0] * dst1[1]);
        let c = dst0[0] * dst0[0] + dst1[0] * dst1[0] + 1.0;
        let inv_f = 1.0 / (a * c - b * b * 0.25);

        let a = a * inv_f;
        let b = b * inv_f;
        let c = c * inv_f;

        // Compute the ellipse's $(s,t)$ bounding box in texture space
        let det = -b * b + 4.0 * a * c;
        let inv_det = 1.0 / det;
        let u_sqrt = (det * c).sqrt();
        let v_sqrt = (det * a).sqrt();
        let s0 = Float::ceil(st[0] - 2.0 * inv_det * u_sqrt) as i32;
        let s1 = Float::floor(st[0] + 2.0 * inv_det * u_sqrt) as i32;
        let t0 = Float::ceil(st[1] - 2.0 * inv_det * v_sqrt) as i32;
        let t1 = Float::floor(st[1] + 2.0 * inv_det * v_sqrt) as i32;

        // Scan over ellipse bound and compute quadratic equation

        let mut sum = T::default();
        let mut sum_wts = 0.0;
        for it in t0..=t1 {
            let tt = it as Float - st[1];
            for is in s0..=s1 {
                let ss = is as Float - st[0];
                // Compute squared radius and filter texel if inside ellipse
                let r2 = a * ss * ss + b * ss * tt + c * tt * tt;
                if r2 < 1.0 {
                    let index = usize::min(
                        (r2 * MIPMAP_WEIGHT_LUT.len() as Float) as usize,
                        MIPMAP_WEIGHT_LUT.len() - 1,
                    );
                    let weight = MIPMAP_WEIGHT_LUT[index];
                    sum = sum + self.texel(level, is, it) * weight;
                    sum_wts += weight;
                }
            }
        }
        return sum * (1.0 / sum_wts);
    }

    pub fn triangle(&self, level: usize, st: &Point2f) -> T {
        let level = usize::clamp(level, 0, self.levels() - 1);
        <F32MIPMapImage as MIPMapImage<T>>::bilerp(
            &self.storage.pyramid[level],
            st,
            self.swrap_mode,
            self.twrap_mode,
        )
    }
}

#[allow(dead_code)]
fn write_spectrum_mipmap_image(path: &str, image: &F32MIPMapImage) -> Result<(), PbrtError> {
    let resolution = image.resolution;
    let w = resolution.0;
    let h = resolution.1;
    let mut img: Vec<Float> = vec![0.0; w * h * 3];

    for i in 0..(w * h) {
        let c = image.data.lookup(i);
        img[3 * i + 0] = c as Float;
        img[3 * i + 1] = c as Float;
        img[3 * i + 2] = c as Float;
    }

    return write_image(
        &path,
        &img,
        &Bounds2i::from(((0, 0), (w as i32, h as i32))),
        &Point2i::new(w as i32, h as i32),
    );
}

pub fn create_float_mipmap(
    resolution: &Point2i,
    data: &[Float],
) -> Result<MIPMap<Float>, PbrtError> {
    let mip = MIPMap::<Float>::new(
        resolution,
        data,
        ImageFilter::Bilinear,
        8.0,
        ImageWrap::Repeat,
        ImageWrap::Clamp,
    );
    return Ok(mip);
}

pub fn create_spectrum_mipmap(
    resolution: &Point2i,
    data: &[RGBSpectrum],
) -> Result<MIPMap<RGBSpectrum>, PbrtError> {
    let mip = MIPMap::<RGBSpectrum>::new(
        resolution,
        data,
        ImageFilter::Bilinear,
        8.0,
        ImageWrap::Repeat,
        ImageWrap::Clamp,
    );
    return Ok(mip);
}
