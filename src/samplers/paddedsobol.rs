use crate::base::camera::CameraSample;
use crate::options::*;
use crate::paramdict::*;
use crate::samplers::*;

use crate::util::base::*;
use crate::util::error::*;
use crate::util::lowdiscrepancy::sobol::sobolmatrices::{SOBOL_MATRICES_32, SOBOL_MATRIX_SIZE};
use crate::util::lowdiscrepancy::*;
use crate::util::profile::*;

#[derive(Debug, PartialEq, Default, Clone)]
pub struct PaddedSobolSampler {
    base: BaseSampler,
    samples_per_pixel: u32,
    seed: u32,
    randomize: RandomizeStrategy,
    pixel: Point2i,
    sample_index: u32,
    dimension: u32,
}

impl PaddedSobolSampler {
    pub fn new(samples_per_pixel: u32, randomize: RandomizeStrategy, seed: u32) -> Self {
        if !is_power_of_2(samples_per_pixel) {
            log::warn!(
                "Sobol samplers with non power-of-two sample counts ({}) are suboptimal.",
                samples_per_pixel
            );
        }

        Self {
            base: BaseSampler::new(samples_per_pixel),
            samples_per_pixel,
            seed,
            randomize,
            pixel: Point2i::zero(),
            sample_index: 0,
            dimension: 0,
        }
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        self.base.start_pixel(p);
        self.pixel = *p;
        self.fill_sample_arrays();
        self.start_pixel_sample(0, 0);
    }

    pub fn get_1d(&mut self) -> Float {
        let _p = ProfilePhase::new(Prof::GetSample);

        let hash = hash_pixel_dimension_seed(&self.pixel, self.dimension, self.seed);
        let index = permutation_element(self.sample_index, self.samples_per_pixel, hash as u32);
        self.dimension += 1;
        sample_dimension_with_hash(0, index as i64, (hash >> 32) as u32, self.randomize)
    }

    pub fn get_2d(&mut self) -> Point2f {
        let _p = ProfilePhase::new(Prof::GetSample);

        let hash = hash_pixel_dimension_seed(&self.pixel, self.dimension, self.seed);
        let index = permutation_element(self.sample_index, self.samples_per_pixel, hash as u32);
        self.dimension += 2;
        Point2f::new(
            sample_dimension_with_hash(0, index as i64, hash as u32, self.randomize),
            sample_dimension_with_hash(1, index as i64, (hash >> 32) as u32, self.randomize),
        )
    }

    pub fn get_pixel_2d(&mut self) -> Point2f {
        self.get_2d()
    }

    pub fn get_camera_sample(&mut self, p_raster: &Point2i) -> CameraSample {
        CameraSample {
            p_film: Point2f::new(p_raster.x as Float, p_raster.y as Float) + self.get_pixel_2d(),
            time: self.get_1d(),
            p_lens: self.get_2d(),
            filter_weight: 1.0,
        }
    }

    pub fn request_1d_array(&mut self, n: u32) {
        self.base.request_1d_array(n);
    }

    pub fn request_2d_array(&mut self, n: u32) {
        self.base.request_2d_array(n);
    }

    pub fn get_1d_array(&mut self, n: u32) -> Option<Vec<Float>> {
        self.base.get_1d_array(n)
    }

    pub fn get_2d_array(&mut self, n: u32) -> Option<Vec<Vector2f>> {
        self.base.get_2d_array(n)
    }

    pub fn start_next_sample(&mut self) -> bool {
        let ok = self.base.start_next_sample();
        if ok {
            self.start_pixel_sample(self.base.current_pixel_sample_index, 0);
        }
        ok
    }

    pub fn set_sample_number(&mut self, sample_num: u32) -> bool {
        let ok = self.base.set_sample_number(sample_num);
        if ok {
            self.start_pixel_sample(sample_num, 0);
        }
        ok
    }

    pub fn get_samples_per_pixel(&self) -> u32 {
        self.samples_per_pixel
    }

    pub fn create(
        params: &ParameterDictionary,
        _full_resolution: Point2i,
    ) -> Result<PaddedSobolSampler, PbrtError> {
        let mut nsamp = params.get_one_int("pixelsamples", 16) as u32;
        let randomize = parse_randomize_strategy(
            params.get_one_string("randomization", "fastowen"),
            "PaddedSobolSampler",
        )?;
        let seed = params.get_one_int("seed", PbrtOptions::get().seed as i32) as u32;
        {
            let options = PbrtOptions::get();
            if options.quick_render {
                nsamp = 1;
            }
        }
        Ok(Self::new(nsamp, randomize, seed))
    }

    pub fn start_pixel_sample(&mut self, sample_index: u32, dim: u32) {
        self.pixel = self.base.current_pixel;
        self.sample_index = sample_index;
        self.dimension = dim;
    }

    fn fill_sample_arrays(&mut self) {
        let mut dim = 5u32;
        for i in 0..self.base.samples1d_array_sizes.len() {
            let total = self.samples_per_pixel * self.base.samples1d_array_sizes[i];
            let hash = hash_pixel_dimension_seed(&self.pixel, dim, self.seed);
            for j in 0..total {
                let index = permutation_element(j, total, hash as u32);
                self.base.sample_array1d[i][j as usize] = sample_dimension_with_hash(
                    0,
                    index as i64,
                    (hash >> 32) as u32,
                    self.randomize,
                );
            }
            dim += 1;
        }
        for i in 0..self.base.samples2d_array_sizes.len() {
            let total = self.samples_per_pixel * self.base.samples2d_array_sizes[i];
            let hash = hash_pixel_dimension_seed(&self.pixel, dim, self.seed);
            for j in 0..total {
                let index = permutation_element(j, total, hash as u32);
                self.base.sample_array2d[i][j as usize] = Point2f::new(
                    sample_dimension_with_hash(0, index as i64, hash as u32, self.randomize),
                    sample_dimension_with_hash(
                        1,
                        index as i64,
                        (hash >> 32) as u32,
                        self.randomize,
                    ),
                );
            }
            dim += 2;
        }
    }
}

fn parse_randomize_strategy(
    value: String,
    sampler_name: &str,
) -> Result<RandomizeStrategy, PbrtError> {
    match value.as_str() {
        "none" => Ok(RandomizeStrategy::None),
        "permutedigits" => Ok(RandomizeStrategy::PermuteDigits),
        "fastowen" => Ok(RandomizeStrategy::FastOwen),
        "owen" => Ok(RandomizeStrategy::Owen),
        _ => Err(PbrtError::error(&format!(
            "{}: unknown randomization strategy {}",
            sampler_name, value
        ))),
    }
}

fn sample_dimension_with_hash(
    dimension: u32,
    index: i64,
    hash: u32,
    randomize: RandomizeStrategy,
) -> Float {
    match randomize {
        RandomizeStrategy::None => sobol_sample(index, dimension, 0),
        RandomizeStrategy::PermuteDigits => sobol_sample(index, dimension, hash as u64),
        RandomizeStrategy::FastOwen => {
            randomized_sobol_sample(index, dimension, fast_owen_scramble, hash)
        }
        RandomizeStrategy::Owen => randomized_sobol_sample(index, dimension, owen_scramble, hash),
    }
}

fn randomized_sobol_sample(
    a: i64,
    dimension: u32,
    randomizer: fn(u32, u32) -> u32,
    seed: u32,
) -> Float {
    let mut a = a;
    let mut v: u32 = 0;
    let mut i = usize::min(
        dimension as usize * SOBOL_MATRIX_SIZE,
        SOBOL_MATRICES_32.len() - 1,
    );
    while a != 0 {
        if (a & 1) != 0 {
            v ^= SOBOL_MATRICES_32[i] as u32;
        }
        a >>= 1;
        i += 1;
        i %= SOBOL_MATRICES_32.len();
    }
    let v = randomizer(v, seed);
    Float::min(
        (v as f64 * 2.3283064365386963e-10f64) as Float,
        FLOAT_ONE_MINUS_EPSILON as Float,
    )
}

fn fast_owen_scramble(mut v: u32, seed: u32) -> u32 {
    v = reverse_bits32(v);
    v ^= v.wrapping_mul(0x3d20adea);
    v = v.wrapping_add(seed);
    v = v.wrapping_mul((seed >> 16) | 1);
    v ^= v.wrapping_mul(0x05526c56);
    v ^= v.wrapping_mul(0x53a22864);
    reverse_bits32(v)
}

fn owen_scramble(mut v: u32, seed: u32) -> u32 {
    if (seed & 1) != 0 {
        v ^= 1u32 << 31;
    }
    for b in 1..32 {
        let mask = (!0u32) << (32 - b);
        if (mix_bits(((v & mask) ^ seed) as u64) as u32 & (1u32 << b)) != 0 {
            v ^= 1u32 << (31 - b);
        }
    }
    v
}

fn hash_pixel_dimension_seed(pixel: &Point2i, dimension: u32, seed: u32) -> u64 {
    let mut buf = [0u8; 16];
    buf[0..4].copy_from_slice(&pixel.x.to_ne_bytes());
    buf[4..8].copy_from_slice(&pixel.y.to_ne_bytes());
    buf[8..12].copy_from_slice(&dimension.to_ne_bytes());
    buf[12..16].copy_from_slice(&seed.to_ne_bytes());
    murmur_hash_64a(&buf, 0)
}

fn murmur_hash_64a(key: &[u8], seed: u64) -> u64 {
    let m = 0xc6a4a7935bd1e995u64;
    let r = 47u32;

    let len = key.len() as u64;
    let mut h = seed ^ len.wrapping_mul(m);

    let nblocks = key.len() / 8;
    for i in 0..nblocks {
        let start = i * 8;
        let mut k = u64::from_ne_bytes(key[start..start + 8].try_into().unwrap());
        k = k.wrapping_mul(m);
        k ^= k >> r;
        k = k.wrapping_mul(m);

        h ^= k;
        h = h.wrapping_mul(m);
    }

    let tail = &key[nblocks * 8..];
    match tail.len() {
        7 => {
            h ^= (tail[6] as u64) << 48;
            h ^= (tail[5] as u64) << 40;
            h ^= (tail[4] as u64) << 32;
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        6 => {
            h ^= (tail[5] as u64) << 40;
            h ^= (tail[4] as u64) << 32;
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        5 => {
            h ^= (tail[4] as u64) << 32;
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        4 => {
            h ^= (tail[3] as u64) << 24;
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        3 => {
            h ^= (tail[2] as u64) << 16;
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        2 => {
            h ^= (tail[1] as u64) << 8;
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        1 => {
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        _ => {}
    }

    h ^= h >> r;
    h = h.wrapping_mul(m);
    h ^= h >> r;
    h
}

fn permutation_element(mut i: u32, l: u32, p: u32) -> u32 {
    let mut w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    loop {
        i ^= p;
        i = i.wrapping_mul(0xe170893d);
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i = i.wrapping_mul(0x0929eb3f);
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i = i.wrapping_mul(1 | (p >> 27));
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

fn mix_bits(mut v: u64) -> u64 {
    v ^= v >> 31;
    v = v.wrapping_mul(0x7fb5d329728ea185);
    v ^= v >> 27;
    v = v.wrapping_mul(0x81dadef4bc2dd44d);
    v ^= v >> 33;
    v
}
