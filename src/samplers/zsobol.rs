use crate::base::camera::CameraSample;
use crate::options::*;
use crate::paramdict::*;
use crate::samplers::*;

use crate::util::base::*;
use crate::util::error::*;
use crate::util::lowdiscrepancy::sobol::sobolmatrices::{SOBOL_MATRICES_32, SOBOL_MATRIX_SIZE};
use crate::util::lowdiscrepancy::*;

#[derive(Debug, PartialEq, Default, Clone)]
pub struct ZSobolSampler {
    base: BaseSampler,
    randomize: RandomizeStrategy,
    seed: u32,
    log2_samples_per_pixel: u32,
    n_base4_digits: u32,
    morton_index: u64,
    dimension: u32,
    pixel: Point2i,
}

impl ZSobolSampler {
    pub fn new(
        samples_per_pixel: u32,
        full_resolution: Point2i,
        randomize: RandomizeStrategy,
        seed: u32,
    ) -> Self {
        if !is_power_of_2(samples_per_pixel) {
            log::warn!(
                "Sobol samplers with non power-of-two sample counts ({}) are suboptimal.",
                samples_per_pixel
            );
        }

        let log2_samples_per_pixel = log2int(samples_per_pixel);
        let effective_samples_per_pixel = 1u32 << log2_samples_per_pixel;
        let res = round_up_pow2(u32::max(full_resolution.x as u32, full_resolution.y as u32));
        let log4_samples_per_pixel = (log2_samples_per_pixel + 1) / 2;
        let n_base4_digits = log2int(res) + log4_samples_per_pixel;

        Self {
            base: BaseSampler::new(effective_samples_per_pixel),
            randomize,
            seed,
            log2_samples_per_pixel,
            n_base4_digits,
            morton_index: 0,
            dimension: 0,
            pixel: Point2i::zero(),
        }
    }

    pub fn start_pixel(&mut self, p: &Point2i) {
        self.base.start_pixel(p);
        self.pixel = *p;
        self.fill_sample_arrays();
        self.start_pixel_sample(0, 0);
    }

    pub fn get_1d(&mut self) -> Float {
        let sample_index = self.get_sample_index();
        self.dimension += 1;
        let sample_hash = hash_dimension_seed(self.dimension, self.seed) as u32;
        sample_dimension_with_hash(0, sample_index as i64, sample_hash, self.randomize)
    }

    pub fn get_2d(&mut self) -> Point2f {
        let sample_index = self.get_sample_index();
        self.dimension += 2;
        let bits = hash_dimension_seed(self.dimension, self.seed);
        let sample_hash = [bits as u32, (bits >> 32) as u32];
        Point2f::new(
            sample_dimension_with_hash(0, sample_index as i64, sample_hash[0], self.randomize),
            sample_dimension_with_hash(1, sample_index as i64, sample_hash[1], self.randomize),
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
        1 << self.log2_samples_per_pixel
    }

    pub fn create(
        params: &ParameterDictionary,
        full_resolution: Point2i,
    ) -> Result<ZSobolSampler, PbrtError> {
        let mut nsamp = params.get_one_int("pixelsamples", 16) as u32;
        let randomize = parse_randomize_strategy(
            params.get_one_string("randomization", "fastowen"),
            "ZSobolSampler",
        )?;
        let seed = params.get_one_int("seed", PbrtOptions::get().seed as i32) as u32;
        {
            let options = PbrtOptions::get();
            if options.quick_render {
                nsamp = 1;
            }
        }
        Ok(Self::new(nsamp, full_resolution, randomize, seed))
    }

    pub fn start_pixel_sample(&mut self, sample_index: u32, dim: u32) {
        self.dimension = dim;
        self.pixel = self.base.current_pixel;
        self.morton_index = (encode_morton2(self.pixel.x as u32, self.pixel.y as u32)
            << self.log2_samples_per_pixel)
            | u64::from(sample_index);
    }

    fn get_sample_index(&self) -> u64 {
        get_sample_index_for(
            self.dimension,
            self.morton_index,
            self.log2_samples_per_pixel,
            self.n_base4_digits,
        )
    }

    fn sample_1d_for(&self, sample_num: u32, dimension: u32) -> Float {
        let morton_index = (encode_morton2(self.pixel.x as u32, self.pixel.y as u32)
            << self.log2_samples_per_pixel)
            | u64::from(sample_num);
        let sample_index = get_sample_index_for(
            dimension,
            morton_index,
            self.log2_samples_per_pixel,
            self.n_base4_digits,
        );
        let sample_hash = hash_dimension_seed(dimension + 1, self.seed) as u32;
        sample_dimension_with_hash(0, sample_index as i64, sample_hash, self.randomize)
    }

    fn sample_2d_for(&self, sample_num: u32, dimension: u32) -> Point2f {
        let morton_index = (encode_morton2(self.pixel.x as u32, self.pixel.y as u32)
            << self.log2_samples_per_pixel)
            | u64::from(sample_num);
        let sample_index = get_sample_index_for(
            dimension,
            morton_index,
            self.log2_samples_per_pixel,
            self.n_base4_digits,
        );
        let bits = hash_dimension_seed(dimension + 2, self.seed);
        Point2f::new(
            sample_dimension_with_hash(0, sample_index as i64, bits as u32, self.randomize),
            sample_dimension_with_hash(1, sample_index as i64, (bits >> 32) as u32, self.randomize),
        )
    }

    fn fill_sample_arrays(&mut self) {
        let mut dim = 5u32;
        for i in 0..self.base.samples1d_array_sizes.len() {
            let total = self.get_samples_per_pixel() * self.base.samples1d_array_sizes[i];
            for j in 0..total {
                self.base.sample_array1d[i][j as usize] = self.sample_1d_for(j, dim);
            }
            dim += 1;
        }
        for i in 0..self.base.samples2d_array_sizes.len() {
            let total = self.get_samples_per_pixel() * self.base.samples2d_array_sizes[i];
            for j in 0..total {
                self.base.sample_array2d[i][j as usize] = self.sample_2d_for(j, dim);
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

fn hash_dimension_seed(dimension: u32, seed: u32) -> u64 {
    let mut buf = [0u8; 8];
    buf[0..4].copy_from_slice(&dimension.to_ne_bytes());
    buf[4..8].copy_from_slice(&seed.to_ne_bytes());
    murmur_hash_64a(&buf, 0)
}

fn get_sample_index_for(
    dimension: u32,
    morton_index: u64,
    log2_samples_per_pixel: u32,
    n_base4_digits: u32,
) -> u64 {
    const PERMUTATIONS: [[u8; 4]; 24] = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 2, 1],
        [0, 3, 1, 2],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 2, 0],
        [1, 3, 0, 2],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 1, 2, 0],
        [3, 1, 0, 2],
        [3, 2, 1, 0],
        [3, 2, 0, 1],
        [3, 0, 2, 1],
        [3, 0, 1, 2],
    ];

    let mut sample_index = 0u64;
    let pow2_samples = (log2_samples_per_pixel & 1) != 0;
    let last_digit = if pow2_samples { 1 } else { 0 };
    for i in (last_digit..n_base4_digits).rev() {
        let digit_shift = 2 * i - if pow2_samples { 1 } else { 0 };
        let digit = ((morton_index >> digit_shift) & 3) as usize;
        let higher_digits = morton_index >> (digit_shift + 2);
        let p = ((mix_bits(higher_digits ^ (0x55555555u64 * u64::from(dimension))) >> 24) % 24)
            as usize;
        sample_index |= u64::from(PERMUTATIONS[p][digit]) << digit_shift;
    }

    if pow2_samples {
        let digit = morton_index & 1;
        sample_index |=
            digit ^ (mix_bits((morton_index >> 1) ^ (0x55555555u64 * u64::from(dimension))) & 1);
    }

    sample_index
}

fn encode_morton2(x: u32, y: u32) -> u64 {
    (left_shift2(u64::from(y)) << 1) | left_shift2(u64::from(x))
}

fn left_shift2(mut x: u64) -> u64 {
    x &= 0xffff_ffff;
    x = (x ^ (x << 16)) & 0x0000ffff0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f0f0f0f0f;
    x = (x ^ (x << 2)) & 0x3333333333333333;
    x = (x ^ (x << 1)) & 0x5555555555555555;
    x
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

fn mix_bits(mut v: u64) -> u64 {
    v ^= v >> 31;
    v = v.wrapping_mul(0x7fb5d329728ea185);
    v ^= v >> 27;
    v = v.wrapping_mul(0x81dadef4bc2dd44d);
    v ^= v >> 33;
    v
}
