//! pbrt-v4 `PMJ02BNSampler` (`samplers.h:367-439`, `samplers.cpp:172-225`).
//! Progressive multi-jittered (0,2)-sequence sampler with blue-noise pixel
//! dimensions. Backed by two precomputed tables that we ship as binary
//! blobs alongside the source:
//!
//! - `data/pmj02bn_samples.bin`: 5×65536×2 u32 — `pmj02bnSamples` from
//!   pbrt-v4's `util/pmj02tables.cpp`. Each (x, y) pair is fixed-point in
//!   [0, 1) and reconstructed as `value * 2^-32`.
//! - `data/bluenoise_textures.bin`: 48×128×128 u16 — `BlueNoiseTextures`
//!   from pbrt-v4's `util/bluenoise.cpp`. Stored as `value / 65535`.
//!
//! The blobs were extracted from the pbrt-v4 sources verbatim. Total embedded
//! data is approximately 4 MB.

use crate::base::camera::CameraSample;
use crate::paramdict::*;
use crate::samplers::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::math::{is_power_of_four, log4_int, round_up_power_of_four};
use crate::util::rng::mix_bits;

// --- pmj02bnSamples (5 × 65536 × 2 u32) ---------------------------------
const N_PMJ02BN_SETS: usize = 5;
const N_PMJ02BN_SAMPLES: usize = 65536;
const PMJ02BN_BYTES: &[u8] = include_bytes!("data/pmj02bn_samples.bin");

fn pmj02bn_sample(set_index: usize, sample_index: usize) -> Point2f {
    let set = set_index % N_PMJ02BN_SETS;
    let idx = sample_index % N_PMJ02BN_SAMPLES;
    let off = ((set * N_PMJ02BN_SAMPLES + idx) * 2) * 4;
    let x = u32::from_le_bytes(PMJ02BN_BYTES[off..off + 4].try_into().unwrap());
    let y = u32::from_le_bytes(PMJ02BN_BYTES[off + 4..off + 8].try_into().unwrap());
    // 0x1p-32 — convert u32 fixed-point in [0, 2^32) to f64 in [0, 1).
    Point2f::new(
        (x as f64 * (1.0 / 4294967296.0)) as Float,
        (y as f64 * (1.0 / 4294967296.0)) as Float,
    )
}

// --- BlueNoiseTextures (48 × 128 × 128 u16) -----------------------------
const BLUENOISE_RESOLUTION: usize = 128;
const N_BLUENOISE_TEXTURES: usize = 48;
const BLUENOISE_BYTES: &[u8] = include_bytes!("data/bluenoise_textures.bin");

fn blue_noise(texture_index: i32, p: Point2i) -> Float {
    let tex = (texture_index.rem_euclid(N_BLUENOISE_TEXTURES as i32)) as usize;
    let x = p.x.rem_euclid(BLUENOISE_RESOLUTION as i32) as usize;
    let y = p.y.rem_euclid(BLUENOISE_RESOLUTION as i32) as usize;
    let off = ((tex * BLUENOISE_RESOLUTION + x) * BLUENOISE_RESOLUTION + y) * 2;
    let v = u16::from_le_bytes(BLUENOISE_BYTES[off..off + 2].try_into().unwrap());
    v as Float / 65535.0
}

// --- Hash + PermutationElement ------------------------------------------

/// pbrt-v4 `MurmurHash64A` (`util/hash.h:35-72`). Standard 64-bit Murmur2
/// variant used by `Hash(...)` to mix multiple arguments.
fn murmur_hash_64a(key: &[u8], seed: u64) -> u64 {
    const M: u64 = 0xc6a4a7935bd1e995u64;
    const R: u32 = 47;
    let len = key.len() as u64;
    let mut h = seed ^ len.wrapping_mul(M);
    let nblocks = key.len() / 8;
    for i in 0..nblocks {
        let s = i * 8;
        let mut k = u64::from_le_bytes(key[s..s + 8].try_into().unwrap());
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }
    let tail = &key[nblocks * 8..];
    if !tail.is_empty() {
        let mut hi = h;
        for (i, b) in tail.iter().enumerate() {
            hi ^= (*b as u64) << (i * 8);
        }
        h = hi.wrapping_mul(M);
    }
    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

/// pbrt-v4 `Hash(pixel, dimension, seed)`. The variadic v4 helper packs its
/// arguments into a byte buffer via `memcpy` and hashes that. r4 inlines
/// the exact byte layout for `(Point2i, i32, i32)` so the resulting hashes
/// match v4 bit-for-bit.
fn hash_pixel_dim_seed(pixel: Point2i, dim: i32, seed: i32) -> u64 {
    let mut buf = [0u8; 16];
    buf[0..4].copy_from_slice(&pixel.x.to_le_bytes());
    buf[4..8].copy_from_slice(&pixel.y.to_le_bytes());
    buf[8..12].copy_from_slice(&dim.to_le_bytes());
    buf[12..16].copy_from_slice(&seed.to_le_bytes());
    murmur_hash_64a(&buf, 0)
}

/// pbrt-v4 `PermutationElement(i, l, p)` (`util/math.h`). Owen-style
/// permutation that maps an index `i ∈ [0, l)` to another index in
/// `[0, l)` under the seed `p`.
fn permutation_element(i: u32, l: u32, p: u32) -> u32 {
    let mut w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    let mut i = i;
    loop {
        i ^= p;
        i = i.wrapping_mul(0xe170893d);
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i = i.wrapping_mul(0x0929eb3f);
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i = i.wrapping_mul(1 | p >> 27);
        i = i.wrapping_mul(0x6935fa69);
        i ^= (i & w) >> 11;
        i = i.wrapping_mul(0x74dcb303);
        i ^= (i & w) >> 2;
        i = i.wrapping_mul(0x9e501cc3);
        i ^= (i & w) >> 2;
        i = i.wrapping_mul(0xc860a3df);
        i &= w;
        i ^= i >> 5;
        if i < l {
            break;
        }
    }
    (i + p) % l
}

// --- PMJ02BNSampler ------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PMJ02BNSampler {
    samples_per_pixel: u32,
    seed: i32,
    pixel_tile_size: u32,
    /// Sorted samples: `pixel_samples[(px + py * pixel_tile_size) * spp + s]`.
    pixel_samples: Vec<Point2f>,
    // Per-sample-state mirrored from pbrt-v4.
    pixel: Point2i,
    sample_index: u32,
    dimension: u32,
}

impl Default for PMJ02BNSampler {
    fn default() -> Self {
        PMJ02BNSampler::new(16, 0)
    }
}

impl PartialEq for PMJ02BNSampler {
    fn eq(&self, other: &Self) -> bool {
        self.samples_per_pixel == other.samples_per_pixel
            && self.seed == other.seed
            && self.pixel == other.pixel
            && self.sample_index == other.sample_index
            && self.dimension == other.dimension
    }
}

impl PMJ02BNSampler {
    pub fn new(samples_per_pixel: u32, seed: i32) -> Self {
        if samples_per_pixel as usize > N_PMJ02BN_SAMPLES {
            log::error!(
                "PMJ02BNSampler only supports up to {} samples per pixel",
                N_PMJ02BN_SAMPLES
            );
        }
        // v4: `pixelTileSize = 1 << (Log4Int(nPMJ02bnSamples) -
        //                            Log4Int(RoundUpPow4(samplesPerPixel)))`.
        let pixel_tile_size = 1u32
            << (log4_int(N_PMJ02BN_SAMPLES as u64)
                - log4_int(round_up_power_of_four(samples_per_pixel as u64)));
        let n_pixel_samples =
            (pixel_tile_size as usize) * (pixel_tile_size as usize) * (samples_per_pixel as usize);
        let mut pixel_samples = vec![Point2f::default(); n_pixel_samples];
        let mut n_stored = vec![0u32; (pixel_tile_size as usize) * (pixel_tile_size as usize)];
        for i in 0..N_PMJ02BN_SAMPLES {
            let p = pmj02bn_sample(0, i);
            let p_scaled = Point2f::new(
                p.x * pixel_tile_size as Float,
                p.y * pixel_tile_size as Float,
            );
            let px = p_scaled.x.floor() as i32;
            let py = p_scaled.y.floor() as i32;
            let pixel_offset = (px + py * pixel_tile_size as i32) as usize;
            if n_stored[pixel_offset] == samples_per_pixel {
                // Bin is full — happens when `samplesPerPixel` is not a
                // power of 4 (extra pmj02bn samples beyond what fits).
                continue;
            }
            let sample_offset =
                pixel_offset * samples_per_pixel as usize + n_stored[pixel_offset] as usize;
            pixel_samples[sample_offset] = Point2f::new(
                p_scaled.x - p_scaled.x.floor(),
                p_scaled.y - p_scaled.y.floor(),
            );
            n_stored[pixel_offset] += 1;
        }
        PMJ02BNSampler {
            samples_per_pixel,
            seed,
            pixel_tile_size,
            pixel_samples,
            pixel: Point2i::default(),
            sample_index: 0,
            dimension: 0,
        }
    }

    pub fn create(params: &ParameterDictionary) -> Result<Self, PbrtError> {
        let spp = params.get_one_int("pixelsamples", 16).max(1) as u32;
        let seed = params.get_one_int("seed", 0);
        if !is_power_of_four(spp as u64) {
            log::warn!(
                "PMJ02BNSampler results are best with power-of-4 samples per pixel (1, 4, 16, 64, ...); got {}",
                spp
            );
        }
        Ok(PMJ02BNSampler::new(spp, seed))
    }

    pub fn samples_per_pixel(&self) -> u32 {
        self.samples_per_pixel
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        self.pixel = *p;
    }

    pub fn start_pixel_sample(&mut self, sample_index: u32, dim: u32) {
        self.sample_index = sample_index;
        // v4: dimension = std::max(2, dim). Dimensions 0/1 are reserved
        // for the pixel sample (GetPixel2D), so client dims start at 2.
        self.dimension = dim.max(2);
    }

    pub fn get_1d(&mut self) -> Float {
        let hash = hash_pixel_dim_seed(self.pixel, self.dimension as i32, self.seed);
        let idx = permutation_element(self.sample_index, self.samples_per_pixel, hash as u32);
        let delta = blue_noise(self.dimension as i32, self.pixel);
        self.dimension += 1;
        let v = (idx as Float + delta) / self.samples_per_pixel as Float;
        v.min(ONE_MINUS_EPSILON)
    }

    pub fn get_2d(&mut self) -> Point2f {
        let mut index = self.sample_index;
        let pmj_instance = (self.dimension / 2) as usize;
        if pmj_instance >= N_PMJ02BN_SETS {
            let hash = hash_pixel_dim_seed(self.pixel, self.dimension as i32, self.seed);
            index = permutation_element(self.sample_index, self.samples_per_pixel, hash as u32);
        }
        let u = pmj02bn_sample(pmj_instance, index as usize);
        // Cranley–Patterson rotation
        let bx = blue_noise(self.dimension as i32, self.pixel);
        let by = blue_noise(self.dimension as i32 + 1, self.pixel);
        let mut ux = u.x + bx;
        let mut uy = u.y + by;
        if ux >= 1.0 {
            ux -= 1.0;
        }
        if uy >= 1.0 {
            uy -= 1.0;
        }
        self.dimension += 2;
        Point2f::new(ux.min(ONE_MINUS_EPSILON), uy.min(ONE_MINUS_EPSILON))
    }

    pub fn get_pixel_2d(&mut self) -> Point2f {
        let px = self.pixel.x.rem_euclid(self.pixel_tile_size as i32) as usize;
        let py = self.pixel.y.rem_euclid(self.pixel_tile_size as i32) as usize;
        let offset = (px + py * self.pixel_tile_size as usize) * self.samples_per_pixel as usize;
        self.pixel_samples[offset + self.sample_index as usize]
    }

    pub fn get_camera_sample(&mut self, p_raster: &Point2i) -> CameraSample {
        CameraSample {
            p_film: Point2f::new(p_raster.x as Float, p_raster.y as Float) + self.get_pixel_2d(),
            time: self.get_1d(),
            p_lens: self.get_2d(),
            filter_weight: 1.0,
        }
    }

    /// pbrt-v4 `Sampler::Clone` produces an independent sampler instance
    /// with the same parameters (used by r4's per-thread parallel render).
    pub fn clone_with_seed(&self, seed: u64) -> Self {
        // Mix the per-thread `seed` into the static `self.seed` so two
        // clones with different seeds produce independent streams.
        let mixed = (self.seed as u64) ^ mix_bits(seed);
        PMJ02BNSampler::new(self.samples_per_pixel, (mixed & 0xffff_ffff) as i32)
    }

    pub fn request_1d_array(&mut self, _n: u32) {}
    pub fn request_2d_array(&mut self, _n: u32) {}
    pub fn get_1d_array(&mut self, _n: u32) -> Option<Vec<Float>> {
        None
    }
    pub fn get_2d_array(&mut self, _n: u32) -> Option<Vec<Vector2f>> {
        None
    }

    pub fn round_count(&self, n: u32) -> u32 {
        n
    }

    // --- pbrt-r4 Sampler enum API compat shims ------------------------------

    /// Set the current sample number, matching the sampler interface:
    /// start_pixel_sample` dispatch calls this after `start_pixel`. Resets
    /// the per-sample dimension counter to 2 (dims 0/1 are reserved for
    /// `get_pixel_2d`).
    pub fn set_sample_number(&mut self, sample_num: u32) -> bool {
        self.sample_index = sample_num;
        self.dimension = 2;
        sample_num < self.samples_per_pixel
    }

    pub fn start_next_sample(&mut self) -> bool {
        self.set_sample_number(self.sample_index + 1)
    }

    pub fn get_samples_per_pixel(&self) -> u32 {
        self.samples_per_pixel
    }
}
